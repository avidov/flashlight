/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/app/asr/augmentation/SoundEffectConfig.h"
#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/common/Flags.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/app/asr/decoder/DecodeMaster.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/app/asr/runtime/runtime.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/ext/plugin/ModulePlugin.h"
#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

using fl::ext::afToVector;
using fl::ext::Serializer;
using fl::lib::fileExists;
using fl::lib::format;
using fl::lib::getCurrentDate;
using fl::lib::join;
using fl::lib::pathsConcat;

using namespace fl::app::asr;

namespace {

void parseCmdLineFlagsWrapper(int argc, char** argv) {
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  // Only new flags are re-serialized. Copy any values from deprecated flags to
  // new flags when deprecated flags are present and corresponding new flags
  // aren't
  handleDeprecatedFlags();
}

// Extra flags for IPL
DEFINE_string(unsup_datadir, "", "datadir for unsupervised lists");
DEFINE_string(
    unsup_train,
    "",
    "comma-separated list of unsupervised training data");
DEFINE_int64(slimIPL_start, 0, "updates to start slimIPL");
DEFINE_string(slimIPL_type, "naive", "naive - relabel then bwd;");
DEFINE_string(slimIPL_cache_path, "", "path to save/load cache");
DEFINE_int64(slimIPL_sup_updates, 1, "number of supervised updates");
DEFINE_int64(slimIPL_unsup_updates, 3, "number of unsupervised updates");
} // namespace

int main(int argc, char** argv) {
  fl::init();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + exec +
      " continue [directory] [flags]\n or " + exec +
      " fork [directory/model] [flags]");

  /* ===================== Parse Options ===================== */
  int runIdx = 1; // current #runs in this path
  std::string runPath; // current experiment path
  std::string reloadPath; // path to model to reload
  std::string runStatus = argv[1];
  int64_t startEpoch = 0;
  int64_t startUpdate = 0;
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }
  if (runStatus == kTrainMode) {
    parseCmdLineFlagsWrapper(argc, argv);
    runPath = FLAGS_rundir;
  } else if (runStatus == kContinueMode) {
    runPath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runIdx, runPath))) {
      ++runIdx;
    }
    reloadPath = getRunFile("model_last.bin", runIdx - 1, runPath);
    LOG(INFO) << "reload path is " << reloadPath;
    std::unordered_map<std::string, std::string> cfg;
    std::string version;
    Serializer::load(reloadPath, version, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }
    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
    parseCmdLineFlagsWrapper(argc, argv);
    auto epoch = cfg.find(kEpoch);
    if (epoch == cfg.end()) {
      LOG(WARNING) << "Did not find epoch to start from, starting from 0.";
    } else {
      startEpoch = std::stoi(epoch->second);
    }
    auto nbupdates = cfg.find(kUpdates);
    if (nbupdates == cfg.end()) {
      LOG(WARNING) << "Did not find #updates to start from, starting from 0.";
    } else {
      startUpdate = std::stoi(nbupdates->second);
    }
  } else if (runStatus == kForkMode) {
    reloadPath = argv[2];
    std::unordered_map<std::string, std::string> cfg;
    std::string version;
    Serializer::load(reloadPath, version, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }

    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    parseCmdLineFlagsWrapper(argc, argv);
    runPath = FLAGS_rundir;
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  if (runPath.empty()) {
    LOG(FATAL) << "'runpath' specified by --rundir, --runname cannot be empty";
  }

  af::setSeed(FLAGS_seed);
  fl::DynamicBenchmark::setBenchmarkMode(FLAGS_fl_benchmark_mode);

  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_enable_distributed) {
    fl::ext::initDistributed(
        FLAGS_world_rank,
        FLAGS_world_size,
        FLAGS_max_devices_per_node,
        FLAGS_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(1.0, true, true);
  }

  int worldRank = fl::getWorldRank();
  int worldSize = fl::getWorldSize();
  bool isMaster = (worldRank == 0);

  FL_LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");
  FL_LOG_MASTER(INFO) << "Experiment path: " << runPath;
  FL_LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;

  // flashlight optim mode
  auto flOptimLevel = FLAGS_fl_optim_mode.empty()
      ? fl::OptimLevel::DEFAULT
      : fl::OptimMode::toOptimLevel(FLAGS_fl_optim_mode);
  fl::OptimMode::get().setOptimLevel(flOptimLevel);
  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    LOG(INFO) << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_fl_optim_mode.empty()) {
      LOG(INFO) << "Mixed precision training enabled with no "
                   "optim mode specified - setting optim mode to O1.";
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
    }
  }

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      // extra goodies
      {kUserName, fl::lib::getEnvVar("USER")},
      {kHostName, fl::lib::getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath}};

  std::vector<std::pair<std::string, std::string>> validTagSets =
      parseValidSets(FLAGS_valid);

  /* ===================== Create Dictionary & Lexicon ===================== */
  auto dictPath = FLAGS_tokens;
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified with "
        " --tokens: \"" +
        dictPath + "\"");
  }
  fl::lib::text::Dictionary tokenDict(dictPath);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry("<" + std::to_string(r) + ">");
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  bool isSeq2seqCrit = FLAGS_criterion == kSeq2SeqTransformerCriterion ||
      FLAGS_criterion == kSeq2SeqRNNCriterion;
  if (isSeq2seqCrit) {
    tokenDict.addEntry(fl::app::asr::kEosToken);
    tokenDict.addEntry(fl::lib::text::kPadToken);
  }

  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  fl::lib::text::Dictionary wordDict;
  fl::lib::text::LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = fl::lib::text::loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = fl::lib::text::createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
  }

  /* ===================== Create Dataset ===================== */
  std::unordered_map<std::string, std::string> plCache;
  std::unordered_map<std::string, std::string> plCacheDump;

  // std::unordered_map<, std::string> plCacheFixedSize;

  fl::lib::audio::FeatureParams featParams(
      FLAGS_samplerate,
      FLAGS_framesizems,
      FLAGS_framestridems,
      FLAGS_filterbanks,
      FLAGS_lowfreqfilterbank,
      FLAGS_highfreqfilterbank,
      FLAGS_mfcccoeffs,
      kLifterParam /* lifterparam */,
      FLAGS_devwin /* delta window */,
      FLAGS_devwin /* delta-delta window */);
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  auto featureRes =
      getFeatureType(FLAGS_features_type, FLAGS_channels, featParams);
  int numFeatures = featureRes.first;
  FeatureType featType = featureRes.second;

  TargetGenerationConfig targetGenConfig(
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      FLAGS_criterion,
      FLAGS_surround,
      isSeq2seqCrit,
      FLAGS_replabel,
      true /* skip unk */,
      FLAGS_usewordpiece /* fallback2LetterWordSepLeft */,
      !FLAGS_usewordpiece /* fallback2LetterWordSepLeft */);

  const auto sfxConf = (FLAGS_sfx_config.empty())
      ? std::vector<sfx::SoundEffectConfig>()
      : sfx::readSoundEffectConfigFile(FLAGS_sfx_config);

  auto inputTransform = inputFeatures(
      featParams,
      featType,
      {FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx},
      sfxConf);
  auto targetTransform = targetFeatures(tokenDict, lexicon, targetGenConfig);
  auto wordTransform = wordFeatures(wordDict);
  int targetpadVal = isSeq2seqCrit
      ? tokenDict.getIndex(fl::lib::text::kPadToken)
      : kTargetPadValue;
  int wordpadVal = kTargetPadValue;
  auto padVal = std::make_tuple(0, targetpadVal, wordpadVal);

  std::vector<std::string> trainSplits = fl::lib::split(",", FLAGS_train, true);
  std::vector<std::string> unsupTrainSplits =
      fl::lib::split(",", FLAGS_unsup_train, true);
  auto trainds = createDataset(
      trainSplits,
      FLAGS_datadir,
      FLAGS_batchsize,
      inputTransform,
      targetTransform,
      wordTransform,
      padVal,
      worldRank,
      worldSize,
      false, // allowEmpty
      FLAGS_batching_strategy,
      FLAGS_batching_max_duration);

  auto unsupTrainds = createDataset(
      unsupTrainSplits,
      FLAGS_unsup_datadir,
      FLAGS_batchsize,
      inputTransform,
      targetTransform,
      wordTransform,
      padVal,
      worldRank,
      worldSize,
      false, // allowEmpty
      FLAGS_batching_strategy,
      FLAGS_batching_max_duration);
  LOG(INFO) << "Sup batches " << trainds->size() << " | Unsup size "
            << unsupTrainds->size();

  std::map<std::string, std::shared_ptr<fl::Dataset>> validds;
  int64_t validBatchSize =
      FLAGS_validbatchsize == -1 ? FLAGS_batchsize : FLAGS_validbatchsize;
  for (const auto& s : validTagSets) {
    validds[s.first] = createDataset(
        {s.second},
        FLAGS_datadir,
        validBatchSize,
        inputTransform,
        targetTransform,
        wordTransform,
        padVal,
        worldRank,
        worldSize,
        true // allowEmpty
    );
  }

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;
  std::shared_ptr<fl::lib::text::LM> lm;
  std::shared_ptr<WordDecodeMaster> dm;

  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
  (void)fl::ext::ModulePlugin(FLAGS_arch);
  if (runStatus == kTrainMode) {
    FL_LOG_MASTER(INFO) << "Loading architecture file from " << FLAGS_arch;
    // Encoder network, works on audio
    network = fl::ext::ModulePlugin(FLAGS_arch).arch(numFeatures, numClasses);
    if (FLAGS_criterion == kCtcCriterion) {
      criterion = std::make_shared<CTCLoss>(scalemode);
    } else if (FLAGS_criterion == kAsgCriterion) {
      criterion =
          std::make_shared<ASGLoss>(numClasses, scalemode, FLAGS_transdiag);
    } else if (FLAGS_criterion == kSeq2SeqRNNCriterion) {
      std::vector<std::shared_ptr<AttentionBase>> attentions;
      for (int i = 0; i < FLAGS_decoderattnround; i++) {
        attentions.push_back(createAttention());
      }
      criterion = std::make_shared<Seq2SeqCriterion>(
          numClasses,
          FLAGS_encoderdim,
          tokenDict.getIndex(fl::app::asr::kEosToken),
          tokenDict.getIndex(fl::lib::text::kPadToken),
          FLAGS_maxdecoderoutputlen,
          attentions,
          createAttentionWindow(),
          FLAGS_trainWithWindow,
          FLAGS_pctteacherforcing,
          FLAGS_labelsmooth,
          FLAGS_inputfeeding,
          FLAGS_samplingstrategy,
          FLAGS_gumbeltemperature,
          FLAGS_decoderrnnlayer,
          FLAGS_decoderattnround,
          FLAGS_decoderdropout);
    } else if (FLAGS_criterion == kSeq2SeqTransformerCriterion) {
      criterion = std::make_shared<TransformerCriterion>(
          numClasses,
          FLAGS_encoderdim,
          tokenDict.getIndex(fl::app::asr::kEosToken),
          tokenDict.getIndex(fl::lib::text::kPadToken),
          FLAGS_maxdecoderoutputlen,
          FLAGS_am_decoder_tr_layers,
          createAttention(),
          createAttentionWindow(),
          FLAGS_trainWithWindow,
          FLAGS_labelsmooth,
          FLAGS_pctteacherforcing,
          FLAGS_am_decoder_tr_dropout,
          FLAGS_am_decoder_tr_layerdrop);
    } else {
      LOG(FATAL) << "unimplemented criterion";
    }
  } else if (runStatus == kForkMode) {
    std::unordered_map<std::string, std::string> cfg; // unused
    std::string version;
    Serializer::load(reloadPath, version, cfg, network, criterion);
    if (version != FL_APP_ASR_VERSION) {
      LOG(WARNING) << "Model version " << version << " and code version "
                   << FL_APP_ASR_VERSION;
    }
  } else { // kContinueMode
    std::unordered_map<std::string, std::string> cfg; // unused
    std::string version;
    Serializer::load(
        reloadPath, version, cfg, network, criterion, netoptim, critoptim);
    if (FLAGS_slimIPL_type == "cache" || FLAGS_slimIPL_type == "pre-cache") {
      LOG(INFO) << "Reading PL cache";
      for (int processIdx = 0; processIdx < worldSize; processIdx++) {
        std::ifstream fcache;
        auto cacheName = getRunFile("model_last_cache", runIdx - 1, runPath) +
            std::to_string(processIdx);
        if (!fileExists(cacheName)) {
          LOG(INFO) << "Read cache from " << cacheName
                    << "; Skip, file doesn't exist";
          continue;
        }
        fcache.open(cacheName);
        std::string line;
        int count = 0;
        while (getline(fcache, line)) {
          auto tmp = fl::lib::split("|", line);
          if (tmp.size() == 0) {
            continue;
          } else if (tmp.size() == 1) {
            plCacheDump[tmp[0]] = "";
          } else {
            plCacheDump[tmp[0]] = tmp[1];
          }
          count++;
        }
        fcache.close();
        LOG(INFO) << "Read cache from " << cacheName
                  << " with number of samples " << count;
      }
      LOG(INFO) << "Reading PL cache is done; total size "
                << plCacheDump.size();
    }
    if (version != FL_APP_ASR_VERSION) {
      LOG(WARNING) << "Model version " << version << " and code version "
                   << FL_APP_ASR_VERSION;
    }
  }
  FL_LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  FL_LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  FL_LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();

  if (!FLAGS_lm.empty()) {
    FL_LOG_MASTER(INFO) << "[Beam-search Decoder] Constructing language model "
                           "and beam search decoder";
    std::vector<float> dummyTransition;
    if (FLAGS_decodertype == "wrd" && FLAGS_lmtype == "kenlm" &&
        FLAGS_criterion == "ctc") {
      lm = std::make_shared<fl::lib::text::KenLM>(FLAGS_lm, wordDict);
      dm = std::make_shared<WordDecodeMaster>(
          network,
          lm,
          dummyTransition,
          true, // usePlugin
          tokenDict,
          wordDict,
          DecodeMasterTrainOptions{.repLabel = int32_t(FLAGS_replabel),
                                   .wordSepIsPartOfToken = FLAGS_usewordpiece,
                                   .surround = FLAGS_surround,
                                   .wordSep = FLAGS_wordseparator,
                                   .targetPadIdx = targetpadVal});
    } else {
      throw std::runtime_error(
          "Other decoders are not supported yet during training");
    }
  }

  if (runStatus == kTrainMode || runStatus == kForkMode) {
    netoptim = initOptimizer(
        {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
    critoptim =
        initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
  }
  FL_LOG_MASTER(INFO) << "[Network Optimizer] " << netoptim->prettyString();
  FL_LOG_MASTER(INFO) << "[Criterion Optimizer] " << critoptim->prettyString();

  double initLinNetlr = FLAGS_linlr >= 0.0 ? FLAGS_linlr : FLAGS_lr;
  double initLinCritlr =
      FLAGS_linlrcrit >= 0.0 ? FLAGS_linlrcrit : FLAGS_lrcrit;
  std::shared_ptr<LinSegCriterion> linseg;
  std::shared_ptr<fl::FirstOrderOptimizer> linNetoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> linCritoptim;
  if (FLAGS_linseg > startUpdate) {
    if (FLAGS_criterion != kAsgCriterion) {
      LOG(FATAL) << "linseg may only be used with ASG criterion";
    }
    linseg = std::make_shared<LinSegCriterion>(numClasses, scalemode);
    linseg->setParams(criterion->param(0), 0);
    FL_LOG_MASTER(INFO) << "[Criterion] " << linseg->prettyString()
                        << " (for first " << FLAGS_linseg - startUpdate
                        << " updates)";

    linNetoptim = initOptimizer(
        {network},
        FLAGS_netoptim,
        initLinNetlr,
        FLAGS_momentum,
        FLAGS_weightdecay);
    linCritoptim =
        initOptimizer({linseg}, FLAGS_critoptim, initLinCritlr, 0.0, 0.0);

    FL_LOG_MASTER(INFO) << "[Network Optimizer] " << linNetoptim->prettyString()
                        << " (for first " << FLAGS_linseg - startUpdate
                        << " updates)";
    FL_LOG_MASTER(INFO) << "[Criterion Optimizer] "
                        << linCritoptim->prettyString() << " (for first "
                        << FLAGS_linseg - startUpdate << " updates)";
  }

  /* ===================== Meters ===================== */
  TrainMeters meters;
  for (const auto& s : validTagSets) {
    meters.valid[s.first] = DatasetMeters();
  }

  // best perf so far on valid datasets
  std::unordered_map<std::string, double> validminerrs;
  for (const auto& s : validTagSets) {
    validminerrs[s.first] = DBL_MAX;
  }

  std::unordered_map<std::string, double> validMinWerWithDecoder;
  std::unordered_map<std::string, double> validWerWithDecoder;
  if (dm) {
    for (const auto& s : validTagSets) {
      validMinWerWithDecoder[s.first] = DBL_MAX;
      validWerWithDecoder[s.first] = DBL_MAX;
    }
  }

  /* ===================== Logging ===================== */
  std::ofstream logFile;
  if (isMaster) {
    fl::lib::dirCreate(runPath);
    logFile.open(getRunFile("log", runIdx, runPath));
    if (!logFile) {
      LOG(FATAL) << "failed to open log file for writing";
    }
    // write config
    std::ofstream configFile(getRunFile("config", runIdx, runPath));
    cereal::JSONOutputArchive ar(configFile);
    ar(CEREAL_NVP(config));
  }

  /* ===================== PL Generator ===================== */
  auto tokenToWord = [&isSeq2seqCrit, &tokenDict](
                         const std::vector<int>& tokens,
                         bool isPrediction) -> std::vector<std::string> {
    std::vector<std::string> letters;
    if (isPrediction) {
      letters = tknPrediction2Ltr(
          tokens,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
    } else {
      letters = tknTarget2Ltr(
          tokens,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
    }
    return tkn2Wrd(letters, FLAGS_wordseparator);
  };

  /* ===================== Hooks ===================== */
  auto logStatus =
      [&logFile, &validTagSets, isMaster](
          TrainMeters& mtrs,
          std::unordered_map<std::string, double>& validWerWithDecoder,
          int64_t epoch,
          int64_t nupdates,
          double lr,
          double lrcrit) {
        syncMeter(mtrs);

        if (isMaster) {
          auto logMsg = getLogString(
              mtrs, validWerWithDecoder, epoch, nupdates, lr, lrcrit);
          FL_LOG_MASTER(INFO) << logMsg;
          appendToLog(logFile, logMsg);
        }
      };

  std::ofstream memLog;
  if (FLAGS_fl_log_mem_ops_interval > 0 && isMaster) {
    auto* curMemMgr =
        fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
    if (curMemMgr) {
      memLog.open(getRunFile("mem", runIdx, runPath));
      if (!memLog) {
        LOG(FATAL) << "failed to open memory log file="
                   << getRunFile("mem", runIdx, runPath) << " for writing";
      }
      curMemMgr->setLogStream(&memLog);
      curMemMgr->setLoggingEnabled(true);
      curMemMgr->setLogFlushInterval(FLAGS_fl_log_mem_ops_interval);
    }
  }

  auto saveModels = [&](int iter, int totalUpdates) {
    if (FLAGS_slimIPL_type == "pre-cache" || FLAGS_slimIPL_type == "cache") {
      std::ofstream fcache;
      auto cacheName = getRunFile("model_last_cache", runIdx, runPath) +
          std::to_string(worldRank);
      fcache.open(cacheName);
      for (auto const& element : plCache) {
        fcache << element.first << "|" << element.second << std::endl;
      }
      fcache.close();
    }
    if (isMaster) {
      // Save last epoch
      config[kEpoch] = std::to_string(iter);
      config[kUpdates] = std::to_string(totalUpdates);

      std::string filename;
      if (FLAGS_itersave) {
        filename =
            getRunFile(format("model_iter_%03d.bin", iter), runIdx, runPath);
        Serializer::save(
            filename,
            FL_APP_ASR_VERSION,
            config,
            network,
            criterion,
            netoptim,
            critoptim);
      }

      // save last model
      filename = getRunFile("model_last.bin", runIdx, runPath);
      Serializer::save(
          filename,
          FL_APP_ASR_VERSION,
          config,
          network,
          criterion,
          netoptim,
          critoptim);

      // save if better than ever for one valid
      for (const auto& v : validminerrs) {
        double verr = meters.valid[v.first].wrdEdit.errorRate()[0];
        if (verr < validminerrs[v.first]) {
          validminerrs[v.first] = verr;
          std::string cleaned_v = cleanFilepath(v.first);
          std::string vfname =
              getRunFile("model_" + cleaned_v + ".bin", runIdx, runPath);
          Serializer::save(
              vfname,
              FL_APP_ASR_VERSION,
              config,
              network,
              criterion,
              netoptim,
              critoptim);
        }
      }

      // save if better than ever for one valid with lm decoding
      for (const auto& v : validMinWerWithDecoder) {
        double verr = validWerWithDecoder[v.first];
        if (verr < validMinWerWithDecoder[v.first]) {
          validMinWerWithDecoder[v.first] = verr;
          std::string cleaned_v = cleanFilepath(v.first);
          std::string vfname = getRunFile(
              "model_" + cleaned_v + "_decoder.bin", runIdx, runPath);
          Serializer::save(
              vfname,
              FL_APP_ASR_VERSION,
              config,
              network,
              criterion,
              netoptim,
              critoptim);
        }
      }
      // print brief stats on memory allocation (so far)
      auto* curMemMgr =
          fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
      if (curMemMgr) {
        curMemMgr->printInfo("Memory Manager Stats", 0 /* device id */);
      }
    }
  };

  auto evalOutput = [&tokenDict, &criterion, &isSeq2seqCrit](
                        const af::array& op,
                        const af::array& target,
                        const af::array& inputSizes,
                        DatasetMeters& mtr) {
    auto batchsz = op.dims(2);
    for (int b = 0; b < batchsz; ++b) {
      auto tgt = target(af::span, b);
      auto viterbipath = afToVector<int>(
          criterion->viterbiPath(op(af::span, af::span, b), inputSizes.col(b)));
      auto tgtraw = afToVector<int>(tgt);

      // Remove `-1`s appended to the target for batching (if any)
      auto labellen = getTargetSize(tgtraw.data(), tgtraw.size());
      tgtraw.resize(labellen);

      // remap actual, predicted targets for evaluating edit distance error

      auto ltrPred = tknPrediction2Ltr(
          viterbipath,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
      auto ltrTgt = tknTarget2Ltr(
          tgtraw,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);

      auto wrdPred = tkn2Wrd(ltrPred, FLAGS_wordseparator);
      auto wrdTgt = tkn2Wrd(ltrTgt, FLAGS_wordseparator);

      mtr.tknEdit.add(ltrPred, ltrTgt);
      mtr.wrdEdit.add(wrdPred, wrdTgt);
    }
  };

  auto test = [&evalOutput, &dm, &lexicon, &isSeq2seqCrit](
                  std::shared_ptr<fl::Module> ntwrk,
                  std::shared_ptr<SequenceCriterion> crit,
                  std::shared_ptr<fl::Dataset> validds,
                  DatasetMeters& mtrs,
                  double& dmErr) {
    ntwrk->eval();
    crit->eval();
    mtrs.tknEdit.reset();
    mtrs.wrdEdit.reset();
    mtrs.loss.reset();

    auto curValidset = loadPrefetchDataset(
        validds, FLAGS_nthread, false /* shuffle */, 0 /* seed */);

    if (dm) {
      fl::TimeMeter timer;
      timer.resume();
      FL_LOG_MASTER(INFO) << "[Beam-search decoder]   * DM: compute emissions";
      auto eds = dm->forward(curValidset);
      FL_LOG_MASTER(INFO) << "[Beam-search decoder]   * DM: decode";
      std::vector<double> lmweights;
      for (double lmweight = FLAGS_lmweight_low;
           lmweight <= FLAGS_lmweight_high;
           lmweight += FLAGS_lmweight_step) {
        lmweights.push_back(lmweight);
      }
      std::vector<std::vector<int64_t>> wordEditDst(lmweights.size());
      std::vector<std::thread> threads;
      for (int i = 0; i < lmweights.size(); i++) {
        threads.push_back(
            std::thread([&lmweights, &wordEditDst, dm, eds, &lexicon, i]() {
              double lmweight = lmweights[i];
              DecodeMasterLexiconOptions opt = {
                  .beamSize = FLAGS_beamsize,
                  .beamSizeToken = FLAGS_beamsizetoken,
                  .beamThreshold = FLAGS_beamthreshold,
                  .lmWeight = lmweight,
                  .silScore = FLAGS_silscore,
                  .wordScore = FLAGS_wordscore,
                  .unkScore = FLAGS_unkscore,
                  .logAdd = FLAGS_logadd,
                  .silToken = FLAGS_wordseparator,
                  .blankToken = kBlankToken,
                  .unkToken = fl::lib::text::kUnkToken,
                  .smearMode =
                      (FLAGS_smearing == "max"
                           ? fl::lib::text::SmearingMode::MAX
                           : fl::lib::text::SmearingMode::NONE)};
              auto pds = dm->decode(eds, lexicon, opt);
              // return token distance and word distance stats
              wordEditDst[i] = dm->computeMetrics(pds).second;
            }));
      }
      for (auto& thread : threads) {
        thread.join();
      }
      dmErr = DBL_MAX;
      for (int i = 0; i < lmweights.size(); i++) {
        af::array currentEditDist =
            af::constant((long long)(wordEditDst[i][0]), af::dim4(1, 1, 1, 1));
        af::array currentTokens =
            af::constant((long long)(wordEditDst[i][1]), af::dim4(1, 1, 1, 1));
        if (FLAGS_enable_distributed) {
          fl::allReduce(currentEditDist);
          fl::allReduce(currentTokens);
        }
        double wer = (double)currentEditDist.scalar<long long>() /
            currentTokens.scalar<long long>() * 100.0;
        FL_LOG_MASTER(INFO)
            << "[Beam-search decoder]   * DM: lmweight=" << lmweights[i]
            << " WER: " << wer;
        dmErr = std::min(dmErr, wer);
      }
      FL_LOG_MASTER(INFO) << "[Beam-search decoder]   * DM: done with best WER "
                          << dmErr;
      timer.stop();
      FL_LOG_MASTER(INFO)
          << "[Beam-search decoder] time spent on grid-search for decoding: "
          << timer.value() << "s";
    }

    for (auto& batch : *curValidset) {
      fl::Variable output = ntwrk
                                ->forward({fl::input(batch[kInputIdx]),
                                           fl::noGrad(batch[kDurationIdx])})
                                .front();
      std::vector<fl::Variable> critArgs = {
          output, fl::Variable(batch[kTargetIdx], false)};
      if (isSeq2seqCrit) {
        critArgs.push_back(fl::Variable(batch[kDurationIdx], false));
        critArgs.push_back(fl::Variable(batch[kTargetSizeIdx], false));
      }
      auto loss = crit->forward(critArgs).front();
      mtrs.loss.add(loss.array());
      evalOutput(output.array(), batch[kTargetIdx], batch[kDurationIdx], mtrs);
    }
  };

  int64_t curEpoch = startEpoch;
  auto train = [&meters,
                &validWerWithDecoder,
                &test,
                &logStatus,
                &saveModels,
                &evalOutput,
                &validds,
                &curEpoch,
                &startUpdate,
                &isSeq2seqCrit,
                &targetTransform,
                &tokenToWord,
                &targetpadVal,
                &plCache,
                &plCacheDump,
                reducer](
                   std::shared_ptr<fl::Module> ntwrk,
                   std::shared_ptr<SequenceCriterion> crit,
                   std::shared_ptr<fl::Dataset> trainset,
                   std::shared_ptr<fl::Dataset> unsupTrainset,
                   std::shared_ptr<fl::FirstOrderOptimizer> netopt,
                   std::shared_ptr<fl::FirstOrderOptimizer> critopt,
                   double initlr,
                   double initcritlr,
                   bool clampCrit,
                   int64_t nbatches) {
    fl::EditDistanceMeter unsupQuality;
    if (reducer) {
      fl::distributeModuleGrads(ntwrk, reducer);
      fl::distributeModuleGrads(crit, reducer);
    }

    meters.train.loss.reset();
    meters.trainUnsup.loss.reset();
    meters.train.tknEdit.reset();
    meters.train.wrdEdit.reset();
    meters.trainUnsup.tknEdit.reset();
    meters.trainUnsup.wrdEdit.reset();

    std::shared_ptr<fl::Module> saug;
    if (FLAGS_saug_start_update >= 0) {
      if (FLAGS_features_type == kFeaturesRaw) {
        saug = std::make_shared<fl::RawWavSpecAugment>(
            FLAGS_filterbanks,
            FLAGS_saug_fmaskf,
            FLAGS_saug_fmaskn,
            FLAGS_saug_tmaskt,
            FLAGS_saug_tmaskp,
            FLAGS_saug_tmaskn,
            FLAGS_filterbanks,
            FLAGS_lowfreqfilterbank,
            FLAGS_highfreqfilterbank,
            FLAGS_samplerate);
      } else {
        saug = std::make_shared<fl::SpecAugment>(
            FLAGS_filterbanks,
            FLAGS_saug_fmaskf,
            FLAGS_saug_fmaskn,
            FLAGS_saug_tmaskt,
            FLAGS_saug_tmaskp,
            FLAGS_saug_tmaskn);
      }
    }

    fl::allReduceParameters(ntwrk);
    fl::allReduceParameters(crit);

    auto resetTimeStatMeters = [&meters]() {
      meters.runtime.reset();
      meters.stats.reset();
      meters.sampletimer.reset();
      meters.fwdtimer.reset();
      meters.critfwdtimer.reset();
      meters.bwdtimer.reset();
      meters.optimtimer.reset();
      meters.timer.reset();
    };
    auto runValAndSaveModel = [&](int64_t totalEpochs,
                                  int64_t totalUpdates,
                                  double lr,
                                  double lrcrit) {
      meters.runtime.stop();
      meters.timer.stop();
      meters.sampletimer.stop();
      meters.fwdtimer.stop();
      meters.critfwdtimer.stop();
      meters.bwdtimer.stop();
      meters.optimtimer.stop();

      // valid
      for (auto& vds : validds) {
        double decodedWer;
        test(ntwrk, crit, vds.second, meters.valid[vds.first], decodedWer);
        if (validWerWithDecoder.find(vds.first) != validWerWithDecoder.end()) {
          validWerWithDecoder[vds.first] = decodedWer;
        }
      }

      // print status
      try {
        logStatus(
            meters, validWerWithDecoder, totalEpochs, totalUpdates, lr, lrcrit);
      } catch (const std::exception& ex) {
        LOG(ERROR) << "Error while writing logs: " << ex.what();
      }
      // save last and best models
      try {
        saveModels(totalEpochs, totalUpdates);
      } catch (const std::exception& ex) {
        LOG(FATAL) << "Error while saving models: " << ex.what();
      }
      // reset meters for next readings
      meters.train.loss.reset();
      meters.train.tknEdit.reset();
      meters.train.wrdEdit.reset();
      meters.trainUnsup.loss.reset();
      meters.trainUnsup.tknEdit.reset();
      meters.trainUnsup.wrdEdit.reset();
    };

    int64_t curBatch = startUpdate;
    double scaleFactor =
        FLAGS_fl_amp_use_mixed_precision ? FLAGS_fl_amp_scale_factor : 1.;
    unsigned int kScaleFactorUpdateInterval =
        FLAGS_fl_amp_scale_factor_update_interval;
    unsigned int kMaxScaleFactor = FLAGS_fl_amp_max_scale_factor;
    unsigned short scaleCounter = 1;
    bool useUnsup = !(unsupTrainset == nullptr);
    FL_LOG_MASTER(INFO) << "Unsup is in use " << useUnsup;
    while (curBatch < nbatches) {
      ++curEpoch; // counts partial epochs too!
      int64_t epochsAfterDecay = curEpoch - FLAGS_lr_decay;
      double lrDecayScale = std::pow(
          0.5,
          (epochsAfterDecay < 0 ? 0
                                : 1 + epochsAfterDecay / FLAGS_lr_decay_step));
      ntwrk->train();
      crit->train();
      if (FLAGS_reportiters == 0) {
        resetTimeStatMeters();
      }
      std::hash<std::string> hasher;
      FL_LOG_MASTER(INFO) << "Shuffling trainset";
      auto curTrainset = loadPrefetchDataset(
          trainset, FLAGS_nthread, true /* shuffle */, curEpoch /* seed */);
      std::shared_ptr<fl::Dataset> curUnsupTrainset;
      if (useUnsup) {
        FL_LOG_MASTER(INFO) << "Shuffling unsup trainset";
        curUnsupTrainset = loadPrefetchDataset(
            unsupTrainset,
            FLAGS_nthread,
            true /* shuffle */,
            curBatch /* seed */);
      }
      af::sync();
      meters.sampletimer.resume();
      meters.runtime.resume();
      meters.timer.resume();
      FL_LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";
      int unsupBatchIdx = 0, supBatchIdx = 0, setsOrderIdx = 0;
      std::vector<bool> setsOrder;
      int unsupSteps = useUnsup ? FLAGS_slimIPL_unsup_updates : 0;
      for (int index = 0; index < FLAGS_slimIPL_sup_updates + unsupSteps;
           index++) {
        if (index < FLAGS_slimIPL_sup_updates) {
          setsOrder.push_back(true);
        } else {
          setsOrder.push_back(false);
        }
      }
      std::random_shuffle(setsOrder.begin(), setsOrder.end());
      while (supBatchIdx < curTrainset->size()) {
        ++curBatch;
        std::vector<af::array> batch;
        bool isSupBatch = setsOrder[setsOrderIdx];
        if (isSupBatch) {
          batch = curTrainset->get(supBatchIdx % curTrainset->size());
          LOG(INFO) << "Sup batch " << curBatch << " | " << supBatchIdx << " | "
                    << batch[kInputIdx].dims();
          ++supBatchIdx;
        } else {
          batch =
              curUnsupTrainset->get(unsupBatchIdx % curUnsupTrainset->size());
          LOG(INFO) << "Unsup batch " << curBatch << " | " << unsupBatchIdx
                    << " | " << batch[kInputIdx].dims();
          ++unsupBatchIdx;
          if (unsupBatchIdx >= curUnsupTrainset->size()) {
            unsupBatchIdx = 0;
            FL_LOG_MASTER(INFO) << "Shuffling unsup trainset";
            curUnsupTrainset = loadPrefetchDataset(
                unsupTrainset,
                FLAGS_nthread,
                true /* shuffle */,
                curBatch /* seed */);
          }
        }
        setsOrderIdx++;
        if (setsOrderIdx >= setsOrder.size()) {
          setsOrderIdx = 0;
          std::random_shuffle(setsOrder.begin(), setsOrder.end());
        }
        double lrScheduleScale;
        if (FLAGS_lrcosine) {
          const double pi = std::acos(-1);
          lrScheduleScale =
              std::cos(((double)curBatch) / ((double)nbatches) * pi / 2.0);
        } else {
          lrScheduleScale =
              std::pow(FLAGS_gamma, (double)curBatch / (double)FLAGS_stepsize);
        }
        netopt->setLr(
            initlr * lrDecayScale * lrScheduleScale *
            std::min(curBatch / double(FLAGS_warmup), 1.0));
        critopt->setLr(
            initcritlr * lrDecayScale * lrScheduleScale *
            std::min(curBatch / double(FLAGS_warmup), 1.0));
        af::sync();
        meters.timer.incUnit();
        meters.sampletimer.stopAndIncUnit();
        if (isSupBatch) {
          meters.stats.add(batch[kDurationIdx], batch[kTargetSizeIdx]);
        }
        if (af::anyTrue<bool>(af::isNaN(batch[kInputIdx])) ||
            af::anyTrue<bool>(af::isNaN(batch[kTargetIdx]))) {
          LOG(FATAL) << "Sample has NaN values - "
                     << join(",", readSampleIds(batch[kSampleIdx]));
        }

        auto predictPL =
            [&](std::vector<af::array> inpBatch) -> std::vector<std::string> {
          ntwrk->eval();
          crit->eval();
          auto outputUnsupOriginal =
              ntwrk
                  ->forward({fl::input(inpBatch[kInputIdx]),
                             fl::noGrad(inpBatch[kDurationIdx])})
                  .front();
          auto viterbiPath = crit->viterbiPath(
              outputUnsupOriginal.array(), inpBatch[kDurationIdx]);

          ntwrk->train();
          crit->train();
          unsupQuality.reset();

          if (curBatch % 100 == 0) {
            FL_LOG_MASTER(INFO)
                << "PL for samples "
                << join(",", readSampleIds(inpBatch[kSampleIdx]));
          }
          std::vector<std::string> plTextArray;
          for (int index = 0; index < viterbiPath.dims(1); index++) {
            auto tokenPrediction = afToVector<int>(viterbiPath.col(index));
            auto plArray = tokenToWord(tokenPrediction, true);
            auto plTrueArray = tokenToWord(
                afToVector<int>(inpBatch[kTargetIdx].col(index)), false);
            auto plText = fl::lib::join(" ", plArray);
            if (curBatch % 100 == 0) {
              FL_LOG_MASTER(INFO) << "PL for index " << index << ": " << plText;
            }
            unsupQuality.add(plArray, plTrueArray);
            plTextArray.push_back(plText);
          }
          fl::ext::syncMeter(unsupQuality);
          if (fl::getWorldRank() == 0) {
            std::cout << "PL Quality for Batch " << curBatch << " : "
                      << unsupQuality.errorRate()[0] << std::endl;
          }
          return plTextArray;
        };

        // Ensure no samples are skipped while adjusting the loss scale factor.
        // When gradient values are Inf/NaN, the model update is skipped and the
        // scale factor is adjusted accordingly for determinism.
        // The AMP algorithm implemented here mirrors:
        // - https://arxiv.org/abs/1710.03740
        // - https://bit.ly/35F5GqX
        // - https://bit.ly/3mn2qr0
        bool retrySample = false;
        bool doUpdate = true;
        do {
          retrySample = false;
          // forward
          meters.fwdtimer.resume();
          auto input = fl::input(batch[kInputIdx]);
          if (FLAGS_saug_start_update >= 0 &&
              curBatch >= FLAGS_saug_start_update) {
            input = saug->forward({input}).front();
          }
          auto output =
              ntwrk->forward({input, fl::noGrad(batch[kDurationIdx])}).front();
          std::vector<fl::Variable> critArgs;
          af::array newUnsupDuration;
          if (isSupBatch) {
            critArgs = {output, fl::Variable(batch[kTargetIdx], false)};
            if (isSeq2seqCrit) {
              critArgs.push_back(fl::Variable(batch[kDurationIdx], false));
              critArgs.push_back(fl::Variable(batch[kTargetSizeIdx], false));
            }
          } else {
            // unsup Batch
            std::vector<std::string> plTextArray;
            std::vector<int> useUnsupSamplesIndices;
            if (FLAGS_slimIPL_type == "naive") {
              // generate by current NN the labels;
              plTextArray = predictPL(batch);
              useUnsupSamplesIndices = std::vector<int>(plTextArray.size());
              std::iota(
                  useUnsupSamplesIndices.begin(),
                  useUnsupSamplesIndices.end(),
                  0);
            } else if (
                FLAGS_slimIPL_type == "cache" ||
                FLAGS_slimIPL_type == "pre-cache") {
              auto samplesIndices = readSampleIds(batch[kSampleIdx]);
              for (int index = 0; index < samplesIndices.size(); index++) {
                if (plCache.find(samplesIndices[index]) == plCache.end() &&
                    plCacheDump.find(samplesIndices[index]) !=
                        plCacheDump.end()) {
                  LOG(INFO)
                      << "Reuse extra loaded cache for sample "
                      << samplesIndices[index] << " for batch " << curBatch;
                  plCache[samplesIndices[index]] =
                      plCacheDump[samplesIndices[index]];
                }
              }

              for (int index = 0; index < samplesIndices.size(); index++) {
                if (plCache.find(samplesIndices[index]) !=
                    plCache.end()) { // place to add filtering too
                  useUnsupSamplesIndices.push_back(index);
                  plTextArray.push_back(plCache[samplesIndices[index]]);
                }
              }
              if (FLAGS_slimIPL_type == "pre-cache" ||
                  useUnsupSamplesIndices.size() == 0) {
                // update Cache before doing model update
                auto plTextArrayPreCache = predictPL(batch);
                for (int index = 0; index < plTextArrayPreCache.size();
                     index++) {
                  plCache[samplesIndices[index]] = plTextArrayPreCache[index];
                }
              }
            }

            if (useUnsupSamplesIndices.size() > 0) {
              af::array maskedSamples = af::array(
                  af::dim4(useUnsupSamplesIndices.size()),
                  useUnsupSamplesIndices.data());
              std::vector<af::array> newTargets, newTargetsSize;
              for (auto& plText : plTextArray) {
                std::vector<char> curTarget(plText.begin(), plText.end());
                auto target = targetTransform(
                    static_cast<void*>(curTarget.data()),
                    {static_cast<dim_t>(curTarget.size())},
                    af::dtype::b8);

                newTargets.push_back(target);
                newTargetsSize.push_back(
                    af::constant(float(target.elements()), 1));
              }
              fl::Dataset::BatchFunction fnc =
                  [targetpadVal](const std::vector<af::array>& arr) {
                    return fl::join(arr, targetpadVal, 1);
                  };
              auto newTargetsSizeBatch = fl::makeBatch(newTargetsSize, nullptr);
              auto newTargetsBatch = fl::makeBatch(newTargets, fnc);
              meters.stats.add(
                  batch[kDurationIdx](maskedSamples), newTargetsSizeBatch);
              // TODO optimize masking early
              critArgs = {
                  output(af::span, af::span, maskedSamples, af::span, true),
                  fl::Variable(newTargetsBatch, false)};
              newUnsupDuration = batch[kDurationIdx](maskedSamples);
              if (isSeq2seqCrit) {
                critArgs.push_back(fl::Variable(newUnsupDuration, false));
                critArgs.push_back(fl::Variable(newTargetsSizeBatch, false));
              }
            } else {
              LOG(INFO)
                  << "Skip unsupervised part of data as PL are not available yet";
            }
          }
          af::sync();
          if (critArgs.size() == 0) {
            doUpdate = false;
            break;
          }
          meters.critfwdtimer.resume();
          auto loss = crit->forward(critArgs).front();
          af::sync();
          meters.fwdtimer.stopAndIncUnit();
          meters.critfwdtimer.stopAndIncUnit();

          if (FLAGS_fl_amp_use_mixed_precision) {
            ++scaleCounter;
            loss = loss * scaleFactor;
          }

          if (af::anyTrue<bool>(af::isNaN(loss.array())) ||
              af::anyTrue<bool>(af::isInf(loss.array()))) {
            if (FLAGS_fl_amp_use_mixed_precision &&
                scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
              scaleFactor = scaleFactor / 2.0f;
              FL_VLOG(2) << "AMP: Scale factor decreased. New value:\t"
                         << scaleFactor;
              scaleCounter = 1;
              retrySample = true;
              continue;
            } else {
              LOG(FATAL) << "Loss has NaN values. Samples - "
                         << join(",", readSampleIds(batch[kSampleIdx]));
            }
          }

          if (hasher(join(",", readSampleIds(batch[kSampleIdx]))) % 100 <=
              FLAGS_pcttraineval) {
            if (isSupBatch) {
              evalOutput(
                  critArgs[0].array(),
                  critArgs[1].array(),
                  batch[kDurationIdx],
                  meters.train);
            } else {
              evalOutput(
                  critArgs[0].array(),
                  critArgs[1].array(),
                  newUnsupDuration,
                  meters.trainUnsup);
            }
          }

          // backward
          meters.bwdtimer.resume();
          netopt->zeroGrad();
          critopt->zeroGrad();
          loss.backward();
          if (reducer) {
            reducer->finalize();
          }
          af::sync();
          meters.bwdtimer.stopAndIncUnit();

          // optimizer
          meters.optimtimer.resume();

          // scale down gradients by batchsize
          af::array totalBatchSizeArr = af::constant(loss.dims(0), 1, f32);
          if (reducer) {
            fl::allReduce(totalBatchSizeArr);
          }
          float totalBatchSize = totalBatchSizeArr.scalar<float>();
          for (const auto& p : ntwrk->params()) {
            if (!p.isGradAvailable()) {
              continue;
            }
            p.grad() = p.grad() / (totalBatchSize * scaleFactor);
            if (FLAGS_fl_amp_use_mixed_precision) {
              if (af::anyTrue<bool>(af::isNaN(p.grad().array())) ||
                  af::anyTrue<bool>(af::isInf(p.grad().array()))) {
                if (scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
                  scaleFactor = scaleFactor / 2.0f;
                  FL_VLOG(2) << "AMP: Scale factor decreased. New value:\t"
                             << scaleFactor;
                  retrySample = true;
                }
                scaleCounter = 1;
                break;
              }
            }
          }
          if (retrySample) {
            meters.optimtimer.stop();
            continue;
          }

          if (isSupBatch) {
            meters.train.loss.add(loss.array());
          } else {
            meters.trainUnsup.loss.add(loss.array());
          }

          for (const auto& p : crit->params()) {
            if (!p.isGradAvailable()) {
              continue;
            }
            p.grad() = p.grad() / (totalBatchSize * scaleFactor);
          }

        } while (retrySample);
        if (doUpdate) {
          // clamp gradients
          if (FLAGS_maxgradnorm > 0) {
            auto params = ntwrk->params();
            if (clampCrit) {
              auto critparams = crit->params();
              params.insert(params.end(), critparams.begin(), critparams.end());
            }
            fl::clipGradNorm(params, FLAGS_maxgradnorm);
          }

          // update weights
          critopt->step();
          netopt->step();
          af::sync();
          if (!isSupBatch) {
            if (FLAGS_slimIPL_type == "cache") {
              auto plTextArray = predictPL(batch);
              auto samplesIndices = readSampleIds(batch[kSampleIdx]);
              for (int index = 0; index < plTextArray.size(); index++) {
                plCache[samplesIndices[index]] = plTextArray[index];
              }
            }
          }

          meters.optimtimer.stopAndIncUnit();

          // update scale factor
          if (FLAGS_fl_amp_use_mixed_precision &&
              scaleFactor < kMaxScaleFactor) {
            if (scaleCounter % kScaleFactorUpdateInterval == 0) {
              scaleFactor *= 2;
              FL_VLOG(2) << "AMP: Scale factor doubled. New value:\t"
                         << scaleFactor;
            } else {
              scaleFactor += 2;
              FL_VLOG(3) << "AMP: Scale factor incremented. New value\t"
                         << scaleFactor;
            }
          }
        } else {
          LOG(INFO) << "Skip update step as unsup data has no label "
                    << curBatch;
        }
        meters.sampletimer.resume();

        if (FLAGS_reportiters > 0 && curBatch % FLAGS_reportiters == 0) {
          runValAndSaveModel(
              curEpoch, curBatch, netopt->getLr(), critopt->getLr());
          resetTimeStatMeters();
          ntwrk->train();
          crit->train();
          meters.sampletimer.resume();
          meters.runtime.resume();
          meters.timer.resume();
        }
        if (curBatch > nbatches) {
          break;
        }
      }
      af::sync();
      if (FLAGS_reportiters == 0) {
        runValAndSaveModel(
            curEpoch, curBatch, netopt->getLr(), critopt->getLr());
      }
    }
  };

  /* ===================== Train ===================== */
  if (FLAGS_linseg - startUpdate > 0) {
    train(
        network,
        linseg,
        trainds,
        unsupTrainds,
        linNetoptim,
        linCritoptim,
        initLinNetlr,
        initLinCritlr,
        false /* clampCrit */,
        FLAGS_linseg - startUpdate);

    startUpdate = FLAGS_linseg;
    FL_LOG_MASTER(INFO) << "Finished LinSeg";
  }

  auto s2s = std::dynamic_pointer_cast<Seq2SeqCriterion>(criterion);
  auto trde = std::dynamic_pointer_cast<TransformerCriterion>(criterion);
  if (FLAGS_pretrainWindow - startUpdate > 0) {
    if (!s2s && !trde) {
      LOG(FATAL) << "Window pretraining only allowed for seq2seq.";
    }
    train(
        network,
        criterion,
        trainds,
        nullptr,
        netoptim,
        critoptim,
        FLAGS_lr,
        FLAGS_lrcrit,
        true,
        FLAGS_pretrainWindow);
    startUpdate = FLAGS_pretrainWindow;
    FL_LOG_MASTER(INFO) << "Finished window pretraining.";
  }
  if (s2s) {
    s2s->clearWindow();
  } else if (trde) {
    trde->clearWindow();
  }

  if (FLAGS_slimIPL_start - startUpdate > 0) {
    train(
        network,
        criterion,
        trainds,
        nullptr,
        netoptim,
        critoptim,
        FLAGS_lr,
        FLAGS_lrcrit,
        true /* clampCrit */,
        FLAGS_slimIPL_start);
    startUpdate = FLAGS_slimIPL_start;
    FL_LOG_MASTER(INFO) << "Finished supervised only pretraining.";
  }

  train(
      network,
      criterion,
      trainds,
      unsupTrainds,
      netoptim,
      critoptim,
      FLAGS_lr,
      FLAGS_lrcrit,
      true /* clampCrit */,
      FLAGS_iter);

  FL_LOG_MASTER(INFO) << "Finished training";
  return 0;
}
