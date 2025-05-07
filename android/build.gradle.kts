// Top-level build file where you can add configuration options common to all sub-projects/modules.
// For an Android library module, this might be the primary build file.

plugins {
    // Apply plugins directly with versions
    id("com.android.library") version "8.4.1"
    id("org.jetbrains.kotlin.android") version "1.9.23"
    id("maven-publish") // Add this later if/when you want to publish to Maven
}

// Apply the plugins to this specific module (the root project in this case)
// apply(plugin = "com.android.library")
// apply(plugin = "org.jetbrains.kotlin.android")

// id("maven-publish") // Add this later if/when you want to publish to Maven

import com.android.build.api.dsl.AndroidSourceDirectorySet

android {
    namespace = "com.cactus.android" // Use the package name we decided on
    compileSdk = 34 // Use a recent SDK version

    defaultConfig {
        minSdk = 24 // Choose your minimum supported Android API level
        // targetSdk = 34 // Usually set in the app module, not mandatory for libraries

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro") // Keep for consumers

        // Specify the ABIs (CPU architectures) to build for / expect in jniLibs
        // This filter also ensures that only these ABIs are included in the AAR
        // if the sourceSets.main.jniLibs.srcDirs is correctly configured (default is src/main/jniLibs)
        ndk {
            abiFilters += listOf(
                "arm64-v8a",
                "x86_64"
                // Add "armeabi-v7a", "x86" if needed, increases AAR size
            )
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false // Ensure this is false
            isShrinkResources = false // Add this to be explicit
            // Comment out ProGuard/R8 application for this build type
            // proguardFiles(
            //     getDefaultProguardFile("proguard-android-optimize.txt"),
            //     "proguard-rules.pro"
            // )
        }
        debug {
            // Debug specific settings if any
        }
    }

    // Specify Java/Kotlin source compatibility
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    // Add CMake configuration
    externalNativeBuild {
        cmake {
            // Specifies the path to your CMakeLists.txt file.
            // Relative to this build.gradle.kts file
            path = file("src/main/CMakeLists.txt") 
            version = "3.22.1" // Specify the CMake version you use (check your setup)
        }
    }
}

dependencies {
    // No core dependencies needed for the library itself usually, 
    // unless you use AndroidX annotations or specific Kotlin libraries.
    // implementation("org.jetbrains.kotlin:kotlin-stdlib:...") // Usually handled by plugin
    
    // Test dependencies (optional)
    // testImplementation("junit:junit:4.13.2")
    // androidTestImplementation("androidx.test.ext:junit:1.1.5")
    // androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}

// Optional: Add publishing configuration here later using the 'maven-publish' plugin
// publishing { ... }

// Add this block at the end of android/build.gradle.kts

// Add task to generate sources JAR (optional but recommended)
// tasks.register<Jar>("androidSourcesJar") {
//     archiveClassifier.set("sources")
//     from(android.sourceSets.getByName("main").java.srcDirs)
//     // Explicitly specify the Kotlin source directory path
//     from("src/main/kotlin")
// }

// Add the publishing block here, outside afterEvaluate
publishing {
    publications {
        create<MavenPublication>("release") {
            // groupId updated for GitHub Packages convention
            groupId = "io.github.cactus-compute"
            artifactId = "cactus-android"
            version = "0.0.7" 

            // Pulls artifacts from the Android 'release' component
            afterEvaluate {
                from(components["release"]) 
            }

            // Minimal POM details (optional but good practice)
            pom {
                name.set("Cactus Android Library") 
                description.set("An example Android library wrapping native code.")
                // Update URL if your repo structure is different
                url.set("https://github.com/cactus-compute/cactus") 
                // Add license, developers, scm if desired
            }
        }
    }
    repositories {
        // Removed mavenLocal()
        // Configure GitHub Packages repository
        maven {
            name = "GitHubPackages"
            // Assumes repository name is 'cactus' under 'cactus-compute'
            url = uri("https://maven.pkg.github.com/cactus-compute/cactus") 
            credentials {
                // Reads credentials from ~/.gradle/gradle.properties
                username = project.findProperty("gpr.user")?.toString()
                password = project.findProperty("gpr.key")?.toString()
            }
        }
    }
}

// Explicitly configure the generated metadata task to depend on the sources jar task
// tasks.named("generateMetadataFileForReleasePublication") {
//     dependsOn(tasks.named("androidSourcesJar"))
// } 