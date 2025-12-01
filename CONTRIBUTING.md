# Contributing to AdvancedGenomics.jl

Thank you for your interest in contributing to AdvancedGenomics.jl! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:

- **Clear title** describing the problem
- **Detailed description** of the issue
- **Steps to reproduce** the bug
- **Expected vs actual behavior**
- **Environment details** (Julia version, OS, GPU if applicable)
- **Minimal reproducible example** if possible

### Suggesting Features

We love new ideas! To suggest a feature:

1. Check if it's already been suggested in Issues
2. Create a new issue with the "enhancement" label
3. Describe the feature and its use case
4. Explain why it would be valuable

### Pull Requests

We actively welcome pull requests!

#### Before You Start

1. **Fork the repository** and create a new branch
2. **Discuss major changes** by opening an issue first
3. **Check existing PRs** to avoid duplication

#### Development Workflow

1. **Clone your fork:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/AdvancedGenomics.git
   cd AdvancedGenomics
   ```

2. **Create a branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install dependencies:**

   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

4. **Make your changes**

5. **Run tests:**

   ```julia
   using Pkg
   Pkg.test("AdvancedGenomics")
   ```

6. **Commit your changes:**

   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

7. **Push to your fork:**

   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

#### Code Standards

- **Style:** Follow Julia style guidelines
- **Documentation:** Add docstrings for all public functions
- **Tests:** Include tests for new functionality
- **Comments:** Use English comments to explain complex logic
- **Performance:** Consider performance implications

#### Commit Message Format

Use clear, descriptive commit messages:

```
Type: Brief description (50 chars max)

Detailed explanation if needed (wrap at 72 chars)
```

Types:

- `Add:` New feature
- `Fix:` Bug fix
- `Refactor:` Code restructuring
- `Docs:` Documentation changes
- `Test:` Test additions/changes
- `Perf:` Performance improvements

### Documentation

Help improve our documentation:

- Fix typos or clarify explanations
- Add examples to docstrings
- Create tutorials in `examples/`
- Improve README or other docs

### Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged!

## 📝 Development Guidelines

### Adding New Features

When adding a new feature:

1. **Design first:** Consider API design and user experience
2. **Implement:** Write clean, efficient code
3. **Document:** Add comprehensive docstrings
4. **Test:** Write unit tests and integration tests
5. **Example:** Add usage example if applicable

### Testing

We use Julia's built-in test framework:

```julia
using Test
using AdvancedGenomics

@testset "Feature Name" begin
    # Your tests here
    @test function_name(input) == expected_output
end
```

### Performance Considerations

- Profile code for bottlenecks
- Use `@inbounds` and `@simd` where safe
- Consider GPU implementations for large-scale operations
- Benchmark against existing implementations

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **Algorithms:** New GWAS or GS methods
- **Deep Learning:** Novel architectures for genomics
- **Performance:** GPU optimizations, parallelization
- **Documentation:** Tutorials, examples, guides
- **Testing:** Increase test coverage
- **Visualization:** New plotting functions
- **IO:** Support for additional file formats

## 📧 Contact

- **Issues:** https://github.com/1958126580/AdvancedGenomics/issues
- **Discussions:** https://github.com/1958126580/AdvancedGenomics/discussions
- **Email:** 1958126580@qq.com

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

All contributors will be acknowledged in our README and release notes.

Thank you for making AdvancedGenomics.jl better!
