set(TARGET_NAME "openvino_sample")

ie_add_sample(NAME openvino_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES format_reader ie_samples_utils)