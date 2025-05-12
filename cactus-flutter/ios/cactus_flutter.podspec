#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint cactus_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'cactus_flutter'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter FFI plugin project.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'

  # Link our custom XCFramework
  s.vendored_frameworks = 'cactus.xcframework'

  # Specify system frameworks and libraries needed by cactus.xcframework
  s.frameworks = 'Accelerate', 'Foundation', 'Metal', 'MetalKit'
  s.libraries = 'c++'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    # Ensure the header search paths include the framework's headers if needed for the plugin's own native code
    # 'HEADER_SEARCH_PATHS' => '"$(PODS_ROOT)/../../ios/cactus.xcframework/ios-arm64/cactus.framework/Headers" "$(PODS_ROOT)/../../ios/cactus.xcframework/ios-arm64_x86_64-simulator/cactus.framework/Headers"',
    # The above HEADER_SEARCH_PATHS might not be necessary if the framework is linked correctly and its headers are module-mapped.
  }
  s.swift_version = '5.0'

  # If your cactus.xcframework itself has other Pod dependencies, list them here:
  # s.dependency 'SomeOtherPod'
end
