# Cactus Flutter Example

A simple chat interface that demonstrates how to use the Cactus Flutter library to run local LLMs on mobile devices.

## Features

- Chat interface with user and assistant messages
- Automatic model downloading
- Stream token generation for real-time responses
- Proper handling of model initialization and cleanup

## Getting Started

This example app uses the Cactus Flutter package to interact with local LLM models. The app will automatically download a small model (SmolLM2-135M) the first time it runs.

### Prerequisites

- Flutter SDK
- iOS or Android device/emulator
- For iOS: Xcode 14+ (for Metal support)
- For Android: Android SDK with NDK

### Running the Example

1. Make sure you have the Flutter SDK installed and set up
2. Clone the Cactus repository
3. Navigate to the flutter_example directory
4. Run `flutter pub get` to install dependencies
5. Connect a device or start an emulator
6. Run `flutter run` to start the app

## Project Structure

- `lib/main.dart` - Entry point for the application
- `lib/home_screen.dart` - Main chat interface
- `lib/components/` - UI components (message bubbles, input field, header)
- `lib/utils/` - Utility functions for model loading and constants

## How It Works

1. The app initializes by downloading and loading a LLM model
2. Users can type messages and send them to the model
3. The model processes the message and generates a response
4. Responses are streamed token by token for a better user experience

## Customization

You can customize this example by:

- Changing the model URL in `utils/model_utils.dart`
- Adjusting the model parameters for different performance characteristics
- Modifying the UI components to match your design requirements

## Notes

- The first run may take some time as it needs to download the model (approximately 300MB)
- GPU acceleration is enabled on iOS devices for better performance

## License

This example is released under the MIT License. See LICENSE file for details. 