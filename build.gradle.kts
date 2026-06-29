plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.dokka) apply false
}

allprojects {
    repositories {
        // The Koog fork (ai.koog:*:0.8.0-SNAPSHOT) is published here; see the README.
        mavenLocal()
        mavenCentral()
    }
}
