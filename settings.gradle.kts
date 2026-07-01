plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.8.0"
}

rootProject.name = "koog-auto-agent-optimization-lib"

include(":koog-agents-optimization")
include(":koog-optimization-examples")
