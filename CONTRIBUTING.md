# Contributing to Cactus

### Code Style

- **C++ Standard**: Use C++20 features where appropriate
- **Formatting**: Follow the existing code style in the project
- **ARM NEON**: When writing SIMD code, ensure proper alignment and use appropriate intrinsics
- **Comments**: Add comments for complex algorithms, especially in kernel implementations

### Performance Considerations

Cactus is a high-performance inference library optimized for ARM processors. When contributing:

1. **Benchmark Your Changes**: Test performance impact, especially for kernel functions
2. **Memory Efficiency**: Minimize memory allocations in hot paths
3. **SIMD Optimization**: Use ARM NEON intrinsics where beneficial
4. **Cache Awareness**: Consider cache line sizes and memory access patterns

### Testing

Before submitting a PR:

1. Ensure all existing tests pass
2. Add tests for new functionality
3. Test on ARM hardware if possible (Apple Silicon, Raspberry Pi, etc.)
4. Verify quantized operations maintain acceptable accuracy

### Pull Request Process

1. **Fork** the repository and create your branch from `main`
2. **Make your changes** following the guidelines above
3. **Sign-off all commits** using DCO
4. **Update documentation** if you change APIs
5. **Open a Pull Request** with a clear title and description

#### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Tested on ARM hardware
- [ ] Benchmarked performance impact

## Checklist
- [ ] All commits are signed-off (DCO)
- [ ] Code follows project style
- [ ] Comments added where necessary
- [ ] Documentation updated if needed
```

### Reporting Issues

When reporting issues, please include:

1. System information (OS, CPU architecture, ARM variant)
2. Cactus version or commit hash
3. Minimal code to reproduce the issue
4. Expected vs actual behavior
5. Any relevant logs or error messages

### Areas of Contribution

We especially welcome contributions in these areas:

- **Kernel Optimizations**: SIMD implementations for ARM architectures
- **Quantization**: Improved quantization techniques (INT8, INT4)
- **Model Support**: Support for additional model architectures
- **NPU Integration**: Apple Neural Engine and other NPU backends
- **Documentation**: Tutorials, examples, and API documentation
- **Testing**: Test coverage and benchmarking infrastructure

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cactus.git
cd cactus

# Setup the environment (installs dependencies and activates venv)
source ./setup
```

You can run these codes directly on M-series Macbooks since they are ARM-based.
Vanilla M3 CPU-only can run Qwen3-600m-INT8 at 60-70 toks/sec.

### Running Tests

```bash
cactus test                              # Run unit tests and benchmarks
cactus test --model <hf-name>            # Use a specific model for tests
cactus test --ios                        # Run tests on connected iPhone
cactus test --android                    # Run tests on connected Android device
```

### Building

```bash
cactus build                 # Build for ARM chips (libcactus.a)
cactus build --apple         # Build for Apple platforms (iOS/macOS)
cactus build --android       # Build for Android
```

### Downloading Models

```bash
cactus download <hf-name>    # Download and convert model weights
cactus run <hf-name>         # Download, build, and run playground
```

### Questions?

If you have questions about contributing, feel free to:

1. Open an issue for discussion
2. Check existing issues and PRs
3. Review the codebase documentation

## License

By contributing to Cactus, you agree that your contributions will be licensed under the same license as the project (check LICENSE file).

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

Thank you for contributing to Cactus! 