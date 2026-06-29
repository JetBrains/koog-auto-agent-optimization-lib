plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.dokka)
    `maven-publish`
}

group = "ai.koog"
version = "0.1.0-SNAPSHOT"

dependencies {
    api(libs.koog)
    api(libs.koog.serialization)
    api(libs.kotlin.serialization)

    testImplementation(kotlin("test"))
    testImplementation(libs.koog.test)
}

kotlin {
    jvmToolchain(21)
    explicitApi()
    compilerOptions {
        optIn.add("ai.koog.agents.optimization.annotations.OptimizationExtensionApi")
    }
}

tasks.test {
    useJUnitPlatform()
}

// Enforce KDoc on the public API
dokka {
    dokkaSourceSets.named("main") {
        reportUndocumented.set(true)
    }
    dokkaPublications.configureEach {
        failOnWarning.set(true)
    }
}

java {
    withSourcesJar()
}

val dokkaJavadocJar by tasks.registering(Jar::class) {
    dependsOn("dokkaGeneratePublicationHtml")
    from(layout.buildDirectory.dir("dokka/html"))
    archiveClassifier.set("javadoc")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            artifact(dokkaJavadocJar)
            pom {
                name.set("Koog Agents Optimization")
                description.set(
                    "Automatic optimization for Koog agents: make an agent optimizable and run " +
                        "optimizers (BootstrapFewShot, MIPRO, GEPA, ACE) over it.",
                )
                url.set("https://github.com/JetBrains/koog-auto-agent-optimization-package")
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                developers {
                    developer {
                        id.set("jetbrains")
                        name.set("JetBrains")
                    }
                }
                scm {
                    url.set("https://github.com/JetBrains/koog-auto-agent-optimization-package")
                }
            }
        }
    }
    // TODO: Target: mavenLocal (default `publishToMavenLocal`).
}
