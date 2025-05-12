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
  s.platform = :ios, '11.0'

  # Tell CocoaPods to use Swift version 5.
  s.swift_version = '5.0'

  # Vendored framework for the C++ library
  s.vendored_frameworks = 'cactus.xcframework'

  # Ensure the plugin links against necessary system frameworks
  s.frameworks = 'Accelerate', 'Foundation', 'Metal', 'MetalKit'

  # Since the underlying library is C++, ensure correct C++ settings
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_ENABLE_MODULES' => 'YES', # Usually YES for frameworks
    # If your C++ code uses exceptions, you might need this:
    # 'GCC_ENABLE_CPP_EXCEPTIONS' => 'YES',
    # If your framework or its dependencies use Objective-C ARC:
    # 'CLANG_ENABLE_OBJC_ARC' => 'YES',
  }
  # If your framework also has Objective-C++ code, you might need this too:
  s.user_target_xcconfig = { 'CLANG_CXX_LIBRARY' => 'libc++' }

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.user_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }

  # If your cactus.xcframework itself has other Pod dependencies, list them here:
  # s.dependency 'SomeOtherPod'
end
