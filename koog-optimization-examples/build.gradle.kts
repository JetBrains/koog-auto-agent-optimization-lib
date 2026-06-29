// Runnable examples for the published :koog-agents-optimization library.
plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

dependencies {
    implementation(project(":koog-agents-optimization"))
    implementation(libs.koog)
    implementation(libs.kotlin.logging)
    runtimeOnly(libs.logback)
}

kotlin {
    jvmToolchain(21)
    compilerOptions {
        allWarningsAsErrors.set(true)
    }
}

application {
    mainClass.set("ai.koog.agents.optimization.examples.OptimizeWeatherAgentExampleKt")
}

tasks.register<JavaExec>("runSupportRouter") {
    group = "application"
    description = "Runs the support-router GEPA example (two optimizable subgraphs)."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("ai.koog.agents.optimization.examples.OptimizeSupportRouterExampleKt")
}
