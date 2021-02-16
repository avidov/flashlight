cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-asr
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Effects.cpp
  ${CMAKE_CURRENT_LIST_DIR}/EffectsChain.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Io.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

target_link_libraries(
  flashlight-app-asr
  PRIVATE
  sox
  )