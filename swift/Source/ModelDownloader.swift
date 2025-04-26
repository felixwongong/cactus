import Foundation

/// Class for downloading and managing model files
public class ModelDownloader {
    /// Default model directory relative to the application support directory
    public static let defaultModelDirectory = "cactus/models"
    
    /// Error types for model downloading
    public enum ModelDownloadError: Error {
        case invalidURL
        case networkError(Error)
        case fileSystemError(Error)
        case downloadCancelled
        case unknownError
    }
    
    /// Progress information for downloads
    public struct DownloadProgress {
        /// Bytes downloaded
        public let bytesDownloaded: Int64
        
        /// Total bytes expected
        public let bytesTotal: Int64
        
        /// Progress as a fraction (0-1)
        public var fractionCompleted: Double {
            guard bytesTotal > 0 else { return 0 }
            return Double(bytesDownloaded) / Double(bytesTotal)
        }
        
        /// Progress as a percentage (0-100)
        public var percentCompleted: Double {
            return fractionCompleted * 100.0
        }
    }
    
    /// Download options
    public struct DownloadOptions {
        /// URL of the model to download
        public let modelURL: URL
        
        /// Name to use for the folder/file
        public let modelFolderName: String
        
        /// Whether to use a temporary file during download
        public let useTempFile: Bool
        
        /// Create download options
        /// - Parameters:
        ///   - modelURL: URL to download from
        ///   - modelFolderName: Name for the folder/file
        ///   - useTempFile: Whether to use a temporary file during download
        public init(modelURL: URL, modelFolderName: String, useTempFile: Bool = true) {
            self.modelURL = modelURL
            self.modelFolderName = modelFolderName
            self.useTempFile = useTempFile
        }
    }
    
    /// The base directory for models
    private let modelDirectoryURL: URL
    
    /// The session used for downloads
    private let session: URLSession
    
    /// Active download tasks
    private var downloadTasks: [URLSessionDownloadTask] = []
    
    /// Pending completion handlers
    private var completionHandlers: [URLSessionDownloadTask: (Result<URL, ModelDownloadError>) -> Void] = [:]
    
    /// Pending progress handlers
    private var progressHandlers: [URLSessionDownloadTask: (DownloadProgress) -> Void] = [:]
    
    /// Create a model downloader
    /// - Parameters:
    ///   - baseDirectory: Directory for storing models (defaults to app support)
    ///   - session: URLSession to use for downloads
    public init(baseDirectory: URL? = nil, session: URLSession = .shared) {
        if let baseDirectory = baseDirectory {
            modelDirectoryURL = baseDirectory
        } else {
            // Use application support directory by default
            let appSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            modelDirectoryURL = appSupportURL.appendingPathComponent(ModelDownloader.defaultModelDirectory, isDirectory: true)
        }
        
        self.session = session
    }
    
    /// Check if a model exists locally
    /// - Parameter modelName: Name of the model
    /// - Returns: Whether the model exists
    public func modelExists(modelName: String) -> Bool {
        let modelURL = modelDirectoryURL.appendingPathComponent(modelName, isDirectory: true)
        return FileManager.default.fileExists(atPath: modelURL.path)
    }
    
    /// Get the path to a local model
    /// - Parameter modelName: Name of the model
    /// - Returns: URL to the model if it exists, nil otherwise
    public func getModelPath(modelName: String) -> URL? {
        let modelURL = modelDirectoryURL.appendingPathComponent(modelName, isDirectory: true)
        return FileManager.default.fileExists(atPath: modelURL.path) ? modelURL : nil
    }
    
    /// Download a model if it doesn't exist locally
    /// - Parameters:
    ///   - options: Download options
    ///   - progressHandler: Optional handler for download progress
    ///   - completionHandler: Handler called when download completes or fails
    /// - Returns: A function that can be called to cancel the download
    @discardableResult
    public func downloadModelIfNotExists(
        options: DownloadOptions,
        progressHandler: ((DownloadProgress) -> Void)? = nil,
        completionHandler: @escaping (Result<URL, ModelDownloadError>) -> Void
    ) -> () -> Void {
        // Check if model already exists
        let modelURL = modelDirectoryURL.appendingPathComponent(options.modelFolderName, isDirectory: true)
        
        if FileManager.default.fileExists(atPath: modelURL.path) {
            // Model already exists, return its path
            completionHandler(.success(modelURL))
            return {}
        }
        
        // Create model directory if needed
        do {
            try FileManager.default.createDirectory(at: modelDirectoryURL, withIntermediateDirectories: true)
        } catch {
            completionHandler(.failure(.fileSystemError(error)))
            return {}
        }
        
        // Start download
        let task = session.downloadTask(with: options.modelURL) { [weak self] (tempURL, response, error) in
            guard let self = self else { return }
            
            defer {
                // Clean up task references
                if let task = self.downloadTasks.firstIndex(of: $0) {
                    self.downloadTasks.remove(at: task)
                    self.progressHandlers.removeValue(forKey: $0)
                    self.completionHandlers.removeValue(forKey: $0)
                }
            }
            
            // Handle errors
            if let error = error {
                let downloadError: ModelDownloadError
                if (error as NSError).domain == NSURLErrorDomain && (error as NSError).code == NSURLErrorCancelled {
                    downloadError = .downloadCancelled
                } else {
                    downloadError = .networkError(error)
                }
                completionHandler(.failure(downloadError))
                return
            }
            
            guard let tempURL = tempURL else {
                completionHandler(.failure(.unknownError))
                return
            }
            
            do {
                // Move file to final location
                try FileManager.default.createDirectory(at: modelURL, withIntermediateDirectories: true)
                let finalURL = modelURL.appendingPathComponent(options.modelURL.lastPathComponent)
                
                // Remove existing file if needed
                if FileManager.default.fileExists(atPath: finalURL.path) {
                    try FileManager.default.removeItem(at: finalURL)
                }
                
                try FileManager.default.moveItem(at: tempURL, to: finalURL)
                completionHandler(.success(finalURL))
            } catch {
                completionHandler(.failure(.fileSystemError(error)))
            }
        }
        
        // Store tasks and handlers
        downloadTasks.append(task)
        if let progressHandler = progressHandler {
            progressHandlers[task] = progressHandler
        }
        completionHandlers[task] = completionHandler
        
        // Start download
        task.resume()
        
        // Return cancel function
        return { [weak self, weak task] in
            guard let task = task, let self = self else { return }
            
            task.cancel()
            
            // Clean up
            if let index = self.downloadTasks.firstIndex(of: task) {
                self.downloadTasks.remove(at: index)
            }
            self.progressHandlers.removeValue(forKey: task)
            
            // Call completion with cancellation
            if let completion = self.completionHandlers.removeValue(forKey: task) {
                completion(.failure(.downloadCancelled))
            }
        }
    }
    
    /// Download model asynchronously (Swift Concurrency version)
    /// - Parameters:
    ///   - options: Download options
    ///   - progressHandler: Optional handler for download progress
    /// - Returns: URL to the downloaded model
    public func downloadModel(
        options: DownloadOptions,
        progressHandler: ((DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        return try await withCheckedThrowingContinuation { continuation in
            let cancelDownload = downloadModelIfNotExists(
                options: options,
                progressHandler: progressHandler
            ) { result in
                switch result {
                case .success(let url):
                    continuation.resume(returning: url)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
            
            // Hold onto the cancel function if needed for cancellation
            // In this implementation, there's no way to cancel from the caller's side
        }
    }
}

/// Session delegate for tracking download progress
extension ModelDownloader: URLSessionDownloadDelegate {
    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        // Report progress to handler if available
        if let progressHandler = progressHandlers[downloadTask] {
            let progress = DownloadProgress(
                bytesDownloaded: totalBytesWritten,
                bytesTotal: totalBytesExpectedToWrite
            )
            DispatchQueue.main.async {
                progressHandler(progress)
            }
        }
    }
    
    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // This is handled in the completion handler of the downloadTask
    }
} 