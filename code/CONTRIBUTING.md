# Contributing to GPU Merge‑Path

Thank you for considering a contribution! This project implements the Parallel
Merge Path algorithm in strict CUDA C, with a focus on correctness, clarity,
and reproducible benchmarking. Contributions that improve performance,
documentation, testing, or usability are welcome.

## How to Contribute

### 1. Fork the repository
Create your own fork and clone it locally.

### 2. Create a feature branch
Use a descriptive name such as:
- feature/shared-memory-optimization
- fix/partition-boundary
- docs/update-readme

### 3. Follow the coding style
- Use strict CUDA C (no C++ features).
- Keep indentation consistent (4 spaces).
- Keep device helpers, kernels, and host wrappers clearly separated.
- Add comments for non-obvious GPU logic.

### 4. Test your changes
Before submitting a pull request:
- Run the small correctness test.
- Run the full benchmarking suite.
- Ensure `results.txt` is generated without errors.
- Confirm GPU output matches CPU output for all tested sizes.

### 5. Update documentation
If your change affects usage, performance, or behavior:
- Update README.md
- Add comments where appropriate

### 6. Submit a Pull Request
Include:
- A clear description of the change
- Why it improves the project
- Any performance impact
- Any new limitations or considerations

## Code of Conduct
Be respectful, constructive, and collaborative. GPU programming is complex —
help others learn and grow.

Thank you for helping improve this project!