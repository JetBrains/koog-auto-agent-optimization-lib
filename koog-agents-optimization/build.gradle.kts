import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import jetbrains.sign.GpgSignSignatoryProvider
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.Base64

buildscript {
    repositories {
        maven { url = uri("https://packages.jetbrains.team/maven/p/jcs/maven") }
        mavenCentral()
    }
    dependencies {
        classpath("com.jetbrains:jet-sign:45.47")
        classpath("com.squareup.okhttp3:okhttp:4.12.0")
    }
}

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.dokka)
    `maven-publish`
    signing
}

group = "ai.koog.optimization"
version = "0.1.0"

dependencies {
    api(libs.koog)
    api(libs.koog.serialization)
    api(libs.kotlin.serialization)

    testImplementation(kotlin("test"))
    testImplementation(libs.koog.test)
    // Already on the test runtime classpath through koog; declared so tests can attach a log appender.
    testImplementation(libs.logback)
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

val artifactsDir = layout.buildDirectory.dir("artifacts/maven")

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
                url.set("https://github.com/JetBrains/koog-auto-agent-optimization-lib")
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
                        organization.set("JetBrains")
                        organizationUrl.set("https://www.jetbrains.com")
                    }
                }
                scm {
                    connection.set("scm:git:https://github.com/JetBrains/koog-auto-agent-optimization-lib.git")
                    url.set("https://github.com/JetBrains/koog-auto-agent-optimization-lib")
                }
            }
        }
    }
    repositories {
        maven {
            name = "artifacts"
            url = uri(artifactsDir)
        }
    }
}

// Maven Central publication, see https://youtrack.jetbrains.com/articles/IJPL-A-611 (do-it-yourself path).
// Signing goes through the JetBrains signing service, which is reachable only from TeamCity.
if (System.getenv("TEAMCITY_VERSION") != null) {
    signing {
        sign(publishing.publications)
        signatories = GpgSignSignatoryProvider()
    }
}

// Clean the local repository first, so that the bundle carries only the version being published
tasks.named("publishMavenPublicationToArtifactsRepository") {
    doFirst { delete(artifactsDir) }
}

val packSonatypeCentralBundle by tasks.registering(Zip::class) {
    group = "publishing"
    dependsOn("publishMavenPublicationToArtifactsRepository")
    from(artifactsDir)
    exclude("**/maven-metadata*")
    archiveFileName.set("bundle.zip")
    destinationDirectory.set(layout.buildDirectory)
}

tasks.register("publishMavenToCentralPortal") {
    group = "publishing"
    dependsOn(packSonatypeCentralBundle)
    val userName = providers.gradleProperty("centralPortalUserName")
    val token = providers.gradleProperty("centralPortalToken")
    val bundle = packSonatypeCentralBundle.flatMap { it.archiveFile }
    val deploymentName = "${project.name}-$version"
    doLast {
        val portal = "https://central.sonatype.com/api/v1/publisher"
        val auth = "Bearer " + Base64.getEncoder().encodeToString("${userName.get()}:${token.get()}".toByteArray())
        val client = OkHttpClient()
        val bundleFile = bundle.get().asFile
        // USER_MANAGED: a validated deployment waits in the Central Portal until someone presses Publish
        val upload = Request.Builder()
            .url("$portal/upload?name=$deploymentName&publishingType=USER_MANAGED")
            .header("Authorization", auth)
            .post(
                MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("bundle", bundleFile.name, bundleFile.asRequestBody())
                    .build(),
            )
            .build()
        val deploymentId = client.newCall(upload).execute().use { response ->
            val body = response.body?.string().orEmpty()
            check(response.code == 201) { "Upload to Central Portal failed with status ${response.code}: $body" }
            body.trim()
        }
        logger.lifecycle("Uploaded deployment $deploymentId, waiting for validation")

        // Validation runs after the upload, so a 201 alone does not mean the bundle is accepted
        val status = Request.Builder()
            .url("$portal/status?id=$deploymentId")
            .header("Authorization", auth)
            .post("".toRequestBody())
            .build()
        val deadline = System.currentTimeMillis() + 30 * 60 * 1000
        while (true) {
            check(System.currentTimeMillis() < deadline) { "Deployment $deploymentId is not validated after 30 minutes" }
            val result = try {
                client.newCall(status).execute().use { response ->
                    val body = response.body?.string().orEmpty()
                    check(response.code == 200) { "Status of deployment $deploymentId: HTTP ${response.code}: $body" }
                    JsonSlurper().parseText(body) as Map<*, *>
                }
            } catch (e: IOException) {
                logger.warn("Status request for deployment $deploymentId failed, retrying: ${e.message}")
                null
            }
            when (val state = result?.get("deploymentState")) {
                "VALIDATED" -> {
                    logger.lifecycle(
                        "Deployment $deploymentId is validated; publish it at https://central.sonatype.com/publishing/deployments",
                    )
                    return@doLast
                }
                "FAILED" -> error("Deployment $deploymentId failed validation: ${JsonOutput.prettyPrint(JsonOutput.toJson(result["errors"]))}")
                else -> if (state != null) logger.lifecycle("Deployment $deploymentId is $state")
            }
            Thread.sleep(10_000)
        }
    }
}
