// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CactusKit",
    // Specify iOS 13.0 as the minimum deployment target
    platforms: [
        .iOS(.v13)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "CactusKit",
            targets: ["CactusKit"]), // The public library product
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.

        // Define the binary target, pointing to the pre-compiled xcframework
        .binaryTarget(
            name: "cactus", // Internal name for the binary target
            path: "Frameworks/cactus.xcframework" // Relative path to the xcframework
        ),

        // Define the main Swift source target
        .target(
            name: "CactusKit",
            dependencies: ["cactus"], // Make CactusKit depend on the binary target "cactus"
            swiftSettings: [
                // Enable C++ interop (adjust flag based on Swift version if needed)
                // Using unsafe flags as this is still experimental
                .unsafeFlags(["-enable-experimental-cxx-interop"], .when(configuration: .debug)),
                .unsafeFlags(["-enable-experimental-cxx-interop"], .when(configuration: .release)),
                // Define the C++ standard to use (e.g., c++17)
                .unsafeFlags(["-cxx-interoperability-mode=default", "-Xcc", "-std=c++17"], .when(configuration: .debug)),
                .unsafeFlags(["-cxx-interoperability-mode=default", "-Xcc", "-std=c++17"], .when(configuration: .release)),
            ],
            linkerSettings: [
                 // Link the C++ standard library explicitly
                .linkedLibrary("c++")
            ]
        ),

        // Define the test target
        .testTarget(
            name: "CactusKitTests",
            dependencies: ["CactusKit"]),
    ]
)
