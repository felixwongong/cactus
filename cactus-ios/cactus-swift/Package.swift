// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CactusKit",
    platforms: [
        .iOS(.v13)
    ],
    products: [
        .library(
            name: "CactusKit",
            targets: ["CactusKit"]), 
    ],
    targets: [
        .binaryTarget(
            name: "cactus", 
            path: "Frameworks/cactus.xcframework" 
        ),

        
        .target(
            name: "CactusKit",
            dependencies: ["cactus"], 
            swiftSettings: [
                .unsafeFlags(["-enable-experimental-cxx-interop"], .when(configuration: .debug)),
                .unsafeFlags(["-enable-experimental-cxx-interop"], .when(configuration: .release)),
                .unsafeFlags(["-cxx-interoperability-mode=default", "-Xcc", "-std=c++17"], .when(configuration: .debug)),
                .unsafeFlags(["-cxx-interoperability-mode=default", "-Xcc", "-std=c++17"], .when(configuration: .release)),
            ],
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),

        .testTarget(
            name: "CactusKitTests",
            dependencies: ["CactusKit"]),
    ]
)
