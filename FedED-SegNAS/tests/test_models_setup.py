#!/usr/bin/env python3
"""
Quick Verification Test for Models Directory Setup
Tests that the models/ directory is properly configured
"""

import sys
import os

def test_directory_exists():
    """Test that models directory exists"""
    models_dir = '/app/FedED-SegNAS/models'
    assert os.path.exists(models_dir), f"❌ Models directory not found at {models_dir}"
    assert os.path.isdir(models_dir), f"❌ {models_dir} is not a directory"
    print("✅ Test 1 PASSED: models/ directory exists")

def test_files_exist():
    """Test that required files exist"""
    base_path = '/app/FedED-SegNAS/models'
    required_files = ['__init__.py', 'README.md', '.gitkeep']
    
    for file in required_files:
        file_path = os.path.join(base_path, file)
        assert os.path.exists(file_path), f"❌ Required file missing: {file}"
        print(f"  ✅ {file} exists")
    
    print("✅ Test 2 PASSED: All required files exist")

def test_module_import():
    """Test that models module can be imported"""
    sys.path.insert(0, '/app/FedED-SegNAS')
    
    try:
        import models
        print("  ✅ models module imported")
        
        # Check version
        assert hasattr(models, '__version__'), "❌ __version__ attribute missing"
        print(f"  ✅ Version: {models.__version__}")
        
        # Check author
        assert hasattr(models, '__author__'), "❌ __author__ attribute missing"
        print(f"  ✅ Author: {models.__author__}")
        
        # Check __all__
        assert hasattr(models, '__all__'), "❌ __all__ attribute missing"
        print(f"  ✅ Exports defined: {len(models.__all__)} items")
        
        print("✅ Test 3 PASSED: Module imports successfully with correct attributes")
        
    except ImportError as e:
        print(f"❌ Test 3 FAILED: Import error - {e}")
        sys.exit(1)

def test_readme_content():
    """Test that README.md has essential content"""
    readme_path = '/app/FedED-SegNAS/models/README.md'
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Check for key sections
    required_sections = [
        'Fuzzy CNN',
        'Federated Learning',
        'Architecture',
        'Usage Example'
    ]
    
    for section in required_sections:
        assert section in content, f"❌ Missing section: {section}"
        print(f"  ✅ Found section: {section}")
    
    print("✅ Test 4 PASSED: README.md has all required sections")

def test_file_sizes():
    """Test that files have reasonable sizes"""
    files_and_sizes = {
        '__init__.py': (500, 5000),  # Between 500 bytes and 5KB
        'README.md': (3000, 10000),   # Between 3KB and 10KB
    }
    
    for filename, (min_size, max_size) in files_and_sizes.items():
        file_path = f'/app/FedED-SegNAS/models/{filename}'
        size = os.path.getsize(file_path)
        
        assert min_size <= size <= max_size, \
            f"❌ {filename} size {size} bytes is outside expected range [{min_size}, {max_size}]"
        
        print(f"  ✅ {filename}: {size} bytes (within range)")
    
    print("✅ Test 5 PASSED: All files have reasonable sizes")

def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("MODELS DIRECTORY VERIFICATION TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("Directory Exists", test_directory_exists),
        ("Required Files", test_files_exist),
        ("Module Import", test_module_import),
        ("README Content", test_readme_content),
        ("File Sizes", test_file_sizes)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}\n")
            failed += 1
    
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Models directory is properly set up.")
        print("✅ Phase 0 is now 100% complete!")
        print("🚀 Ready to begin Phase 2 (Fuzzy CNN implementation)\n")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review and fix.\n")
        return 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
