// import 'dart:io';

// import 'package:path_provider/path_provider.dart'; 

typedef OnInitProgress = void Function(double? progress, String statusMessage, bool isError);

class CactusInitParams {
  final String? modelPath; 
  final String? modelUrl;
  final String? modelFilename; 

  final String? chatTemplate; 
  final int nCtx;
  final int nBatch;
  final int nUbatch;
  final int nGpuLayers;
  final int nThreads;
  final bool useMmap;
  final bool useMlock;
  final bool embedding;
  final int poolingType; 
  final int embdNormalize;
  final bool flashAttn;
  final String? cacheTypeK;
  final String? cacheTypeV;
  
  final OnInitProgress? onInitProgress; 

  CactusInitParams({
    this.modelPath,
    this.modelUrl,
    this.modelFilename,
    this.chatTemplate,
    this.nCtx = 512,
    this.nBatch = 512,
    this.nUbatch = 512,
    this.nGpuLayers = 0, 
    this.nThreads = 4,   
    this.useMmap = true,
    this.useMlock = false,
    this.embedding = false,
    this.poolingType = 0, 
    this.embdNormalize = 1, 
    this.flashAttn = false,
    this.cacheTypeK,
    this.cacheTypeV,
    this.onInitProgress,
  }) {
    if (modelPath == null && modelUrl == null) {
      throw ArgumentError('Either modelPath or modelUrl must be provided.');
    }
    if (modelPath != null && modelUrl != null) {
      throw ArgumentError('Cannot provide both modelPath and modelUrl. Choose one.');
    }
  }
} 