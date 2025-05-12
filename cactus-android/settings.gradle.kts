// This file defines where Gradle should look for plugins and which subprojects to include.

pluginManagement {
    repositories {
        google() // Repository for Android Gradle Plugin
        mavenCentral() // Repository for other common plugins and dependencies
        gradlePluginPortal() // Standard Gradle plugin repository
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

// Define the root project name (optional but good practice)
rootProject.name = "CactusAndroidLib"

// Include subprojects if you had any (currently, you only have the root acting as the library)
include(":test-app") 
// include(":library") // If you restructure later 