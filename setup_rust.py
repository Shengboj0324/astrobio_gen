#!/usr/bin/env python3
"""
Setup script for Rust extensions
================================

This script builds and installs the Rust extensions for the astrobiology AI platform.
It provides high-performance datacube processing with 10-20x speedup over NumPy.

Usage:
    python setup_rust.py build_ext --inplace
    python setup_rust.py install
    
Requirements:
    - Rust toolchain (rustc, cargo)
    - Python development headers
    - NumPy development headers
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# Check for required dependencies
def check_rust_installation():
    """Check if Rust is installed and available"""
    try:
        result = subprocess.run(['rustc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Rust found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Rust not found")
            return False
    except FileNotFoundError:
        print("❌ Rust not found")
        return False

def install_rust():
    """Install Rust if not available"""
    print("🦀 Installing Rust...")
    try:
        # Download and run rustup installer
        if sys.platform.startswith('win'):
            # Windows
            subprocess.run([
                'powershell', '-Command',
                'Invoke-WebRequest -Uri https://win.rustup.rs/ -OutFile rustup-init.exe; ./rustup-init.exe -y'
            ], check=True)
        else:
            # Unix-like systems
            subprocess.run([
                'curl', '--proto', '=https', '--tlsv1.2', '-sSf', 
                'https://sh.rustup.rs', '|', 'sh', '-s', '--', '-y'
            ], shell=True, check=True)
        
        # Source the environment
        if not sys.platform.startswith('win'):
            os.environ['PATH'] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ.get('PATH', '')}"
        
        print("✅ Rust installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Rust: {e}")
        return False

class RustExtension(Extension):
    """Custom extension for Rust modules"""
    
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class BuildRustExtension(build_ext):
    """Custom build command for Rust extensions"""
    
    def run(self):
        # Check for Rust installation
        if not check_rust_installation():
            print("🦀 Rust not found. Installing...")
            if not install_rust():
                raise RuntimeError("Failed to install Rust")
        
        # Check for required Python packages
        self.check_python_dependencies()
        
        # Build Rust extensions
        for ext in self.extensions:
            self.build_rust_extension(ext)
    
    def check_python_dependencies(self):
        """Check for required Python dependencies"""
        required_packages = [
            'numpy',
            'torch',
            'pyo3-setuptools-rust'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing Python packages: {missing_packages}")
            print("Installing missing packages...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
    
    def build_rust_extension(self, ext):
        """Build a single Rust extension"""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Ensure the extension directory exists
        os.makedirs(extdir, exist_ok=True)
        
        # Build configuration
        cfg = 'debug' if self.debug else 'release'
        
        # Build command
        build_args = [
            'cargo', 'build',
            '--manifest-path', os.path.join(ext.sourcedir, 'rust_modules', 'Cargo.toml'),
            '--target-dir', os.path.join(ext.sourcedir, 'target'),
        ]
        
        if cfg == 'release':
            build_args.append('--release')
        
        # Set environment variables
        env = os.environ.copy()
        env['CARGO_TARGET_DIR'] = os.path.join(ext.sourcedir, 'target')
        
        print(f"🔨 Building Rust extension: {ext.name}")
        print(f"   Command: {' '.join(build_args)}")
        
        try:
            subprocess.check_call(build_args, env=env, cwd=ext.sourcedir)
            print(f"✅ Successfully built Rust extension: {ext.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build Rust extension: {e}")
            raise

class InstallWithRust(install):
    """Custom install command that builds Rust extensions first"""
    
    def run(self):
        # Build Rust extensions first
        self.run_command('build_ext')
        
        # Then run normal install
        install.run(self)

def get_version():
    """Get version from Cargo.toml"""
    cargo_toml = Path('rust_modules/Cargo.toml')
    if cargo_toml.exists():
        with open(cargo_toml, 'r') as f:
            for line in f:
                if line.startswith('version = '):
                    return line.split('"')[1]
    return "0.1.0"

def main():
    """Main setup function"""
    
    # Check if we're in the right directory
    if not Path('rust_modules').exists():
        print("❌ rust_modules directory not found")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    setup(
        name='astrobio-rust',
        version=get_version(),
        author='Astrobiology AI Platform Team',
        author_email='team@astrobiology-ai.com',
        description='High-performance Rust extensions for astrobiology AI',
        long_description=open('README.md').read() if Path('README.md').exists() else '',
        long_description_content_type='text/markdown',
        url='https://github.com/astrobiology-ai/platform',
        
        # Python packages
        packages=['rust_integration'],
        package_dir={'rust_integration': 'rust_integration'},
        
        # Rust extensions
        ext_modules=[
            RustExtension('astrobio_rust', sourcedir='.')
        ],
        
        # Custom commands
        cmdclass={
            'build_ext': BuildRustExtension,
            'install': InstallWithRust,
        },
        
        # Dependencies
        install_requires=[
            'numpy>=1.20.0',
            'torch>=1.12.0',
            'pyo3-setuptools-rust>=0.12.0',
        ],
        
        # Python version requirement
        python_requires='>=3.8',
        
        # Classifiers
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Rust',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        
        # Entry points
        entry_points={
            'console_scripts': [
                'astrobio-rust-test=rust_integration.utils:main',
            ],
        },
        
        # Include additional files
        include_package_data=True,
        zip_safe=False,
    )

if __name__ == '__main__':
    main()
