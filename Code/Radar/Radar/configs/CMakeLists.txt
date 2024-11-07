# Specify the path to the configuration file
set(CONFIG_FILE ${CMAKE_SOURCE_DIR}/configs/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg)

# Define the location to copy the configuration file to (e.g., build directory)
set(DESTINATION_DIR ${CMAKE_BINARY_DIR})

# Add a custom command to copy the file
add_custom_command(
    OUTPUT ${DESTINATION_DIR}/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg
    COMMAND ${CMAKE_COMMAND} -E copy ${CONFIG_FILE} ${DESTINATION_DIR}
    DEPENDS ${CONFIG_FILE}
)

# Ensure the file is copied whenever the target is built
add_custom_target(copy_config ALL DEPENDS ${DESTINATION_DIR}/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg)
