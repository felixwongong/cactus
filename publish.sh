#!/usr/bin/env bash
set -euo pipefail

# ./scripts/build-ios.sh
# ./scripts/build-react-android.sh
# ./scripts/build-flutter-android.sh

rm -rf flutter/android/jniLibs.zip
rm -rf flutter/android/src/main/jniLibs/x86_64 
rm -rf react/android/src/main/jniLibs/x86_64

./scripts/build-react.sh
./scripts/build-flutter.sh

git add .
git commit -m "chore: publish"
git push origin main

cd react
yarn release 

cd ..
cd flutter
flutter pub publish 