// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "snowflake-arctic-embed-s-swift",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "ArcticEmbedS",
            targets: ["ArcticEmbedS"]
        ),
        .executable(
            name: "arctic-embed",
            targets: ["arctic-embed"]
        ),
    ],
    targets: [
        .target(
            name: "ArcticEmbedS",
            swiftSettings: [
                .swiftLanguageMode(.v5),
            ]
        ),
        .executableTarget(
            name: "arctic-embed",
            dependencies: ["ArcticEmbedS"],
            swiftSettings: [
                .swiftLanguageMode(.v5),
            ]
        ),
        .testTarget(
            name: "ArcticEmbedSTests",
            dependencies: ["ArcticEmbedS"],
            exclude: ["data"],
            swiftSettings: [
                .swiftLanguageMode(.v5),
            ]
        ),
    ]
)
