import 'package:flutter/material.dart';
import 'package:cactus_flutter/cactus_flutter.dart';
import 'home_screen.dart';

void main() {
  runApp(const CactusChatApp());
}

class CactusChatApp extends StatelessWidget {
  const CactusChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus Chat',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.green,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
} 