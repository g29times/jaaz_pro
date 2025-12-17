# Codebase Cleanup & Organization Plan

**Date**: December 17, 2025
**Status**: Ready for Execution
**Goal**: Clean up redundancy, organize for future development

---

## 🎯 Cleanup Objectives

1. Remove outdated documentation that references old architectures
2. Establish proper .gitignore to prevent tracking generated files
3. Secure sensitive configuration data
4. Remove redundant/unused components
5. Consolidate documentation into clear, current structure
6. Prepare codebase for future Phase 4+ development

---

## 📋 Execution Plan (3 Phases)

### Phase 1: Critical Cleanup (Do First) ⚠️

**Estimated Time**: 15 minutes
**Risk**: Low
**Impact**: High (security, cleanliness)

#### 1.1 Create .gitignore
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Test outputs
test_output_*.png
test_output_*.mmd
test_output_*.jpg
demo_*.png
demo_*.mmd
example_*.png
example_*.mmd

# Logs
*.log
sketch_to_mermaid*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Config with secrets
config.json

# Jupyter
.ipynb_checkpoints/

# Coverage reports
htmlcov/
.coverage
.pytest_cache/
EOF
```

#### 1.2 Secure Configuration
```bash
# Create template config
cp config.json config.template.json

# Edit config.template.json to replace API keys with placeholders
# Then add config.json to .gitignore (already in above)

# Create a note in config.template.json:
# "Copy this to config.json and add your API keys"
```

#### 1.3 Remove Generated Files
```bash
# Remove test outputs
rm -f test_output_*.png test_output_*.mmd sketch_to_mermaid.log

# Remove system files
find . -name ".DS_Store" -delete

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
```

#### 1.4 Delete Outdated Documentation
```bash
# These docs reference old architectures (Vertex AI, local Stable Diffusion)
rm -f IMPLEMENTATION_PLAN.md
rm -f IMAGE_GENERATION_USAGE.md
rm -f example_imagen_generation.py

# Archive Phase 2 plan (already complete)
mv PHASE_2_PLAN.md archive/PHASE_2_PLAN_archived.md 2>/dev/null || mkdir -p archive && mv PHASE_2_PLAN.md archive/
```

**Expected Outcome**: ~3.5MB freed, no sensitive data in git, clean workspace

---

### Phase 2: Component Investigation & Cleanup (High Priority) 🔍

**Estimated Time**: 30 minutes
**Risk**: Medium (requires code review)
**Impact**: High (code clarity, maintainability)

#### 2.1 Check Component Usage

**intelligent_mermaid_generator.py** - Investigate if used:
```bash
grep -r "IntelligentMermaidGenerator" --include="*.py" . | grep -v "__pycache__"
grep -r "intelligent_mermaid_generator" --include="*.py" . | grep -v "__pycache__"
```
- **If NO matches**: DELETE (it's unused legacy code)
- **If matches found**: Review if overlaps with `generators.py`

**imagen_client.py** - Check if Vertex AI client still needed:
```bash
grep -r "ImagenClient" --include="*.py" . | grep -v "__pycache__"
grep -r "imagen_client" --include="*.py" . | grep -v "__pycache__"
```
- **If NO matches**: DELETE (replaced by Gemini native image gen)
- **If matches found**: Consider removing if Gemini native works well

**vlm_engine.py** - Check if legacy vLLM code:
```bash
grep -r "VLMEngine" --include="*.py" . | grep -v "__pycache__"
grep -r "vlm_engine" --include="*.py" . | grep -v "__pycache__"
```
- **If in simple_pipeline.py only**: May be legacy, check if can use Gemini directly
- **If widely used**: Keep but document purpose

#### 2.2 Image Preprocessing Consolidation

**Compare image_processor.py vs image_input_handler.py:**
```bash
# Check which is actually used
grep -r "ImageProcessor\|image_processor" --include="*.py" . | grep -v "__pycache__"
grep -r "ImageInputHandler\|image_input_handler" --include="*.py" . | grep -v "__pycache__"
```

**Decision Tree:**
- If both used for different purposes → Keep both, add comments clarifying roles
- If overlap → Consolidate into one, prefer `image_input_handler.py` (newer)
- If one unused → Delete the unused one

#### 2.3 Remove Unused Components

Based on investigation results:
```bash
# Example (only if confirmed unused):
# rm whiteboard_pipeline/components/intelligent_mermaid_generator.py
# rm whiteboard_pipeline/components/imagen_client.py
```

**Expected Outcome**: ~50KB freed, clearer component structure

---

### Phase 3: Documentation Organization (Medium Priority) 📚

**Estimated Time**: 45 minutes
**Risk**: Low
**Impact**: Medium (clarity for users/developers)

#### 3.1 Final Documentation Structure

**Keep (Current & Accurate):**
```
README.md                      # Main entry point (needs update)
PHASE_2_3_COMPLETE.md         # Completion summary
PROJECT_STATUS.md             # Detailed status
DEVELOPMENT_SUMMARY.md        # Work log
```

**Update (Needs Changes):**
```
QUICK_START.md                # Add Gemini setup instructions
GOOGLE_AI_INTEGRATION.md      # Remove Vertex AI Imagen references
```

**Archive (Historical):**
```
archive/
  └── PHASE_2_PLAN_archived.md  # Moved in Phase 1
```

#### 3.2 Update QUICK_START.md

Add section:
```markdown
## Google Gemini Setup

1. Get API key: https://aistudio.google.com/app/apikey
2. Copy config.template.json to config.json
3. Add your API key to config.json:
   {
     "mermaid_generator": {
       "gemini_api_key": "YOUR_KEY_HERE"
     }
   }
4. Run tests: python3 test_gemini_integration.py
```

#### 3.3 Update GOOGLE_AI_INTEGRATION.md

Remove:
- All references to Vertex AI Imagen
- Image generation setup for Vertex AI

Add:
- Gemini 2.5 Flash Image native generation
- "nano banana" feature explanation
- Updated examples using `generate_diagram_image()`

#### 3.4 Update README.md

Simplify to:
- Current architecture (Phases 1-3 complete)
- Quick start with Gemini
- 10 test descriptions
- Usage examples
- Remove outdated roadmap items

**Expected Outcome**: Clear, accurate documentation for new users

---

## 📊 Summary of Changes

### Files to Delete (Confirmed)

**Documentation:**
- IMPLEMENTATION_PLAN.md
- IMAGE_GENERATION_USAGE.md
- example_imagen_generation.py

**Generated Files:**
- test_output_*.png (4 files)
- test_output_*.mmd (4 files)
- sketch_to_mermaid.log
- .DS_Store (all)
- __pycache__/ (all)

### Files to Investigate & Potentially Delete

**Components:**
- intelligent_mermaid_generator.py (22KB)
- imagen_client.py (7.7KB)
- One of: image_processor.py OR image_input_handler.py (if redundant)

### Files to Update

**Documentation:**
- README.md (major simplification)
- QUICK_START.md (add Gemini setup)
- GOOGLE_AI_INTEGRATION.md (remove Vertex AI)

**Configuration:**
- Create config.template.json
- Secure config.json (add to .gitignore)

### Files to Create

- .gitignore (critical)
- archive/ directory (for historical docs)

---

## 🎯 Expected Outcomes

### Space Savings
- **Immediate**: ~3.5 MB
- **After investigation**: ~4 MB total
- **Files removed from tracking**: ~15,400+ .pyc files

### Code Quality Improvements
- ✅ No sensitive data in git
- ✅ No generated files tracked
- ✅ Clear component structure
- ✅ Current, accurate documentation
- ✅ Clean workspace for future development

### Maintainability Gains
- Easier onboarding for new developers
- Less confusion about what's current
- Clearer architecture
- Better security practices

---

## 🚀 Execution Order (Recommended)

```bash
# 1. Critical Cleanup (15 min)
./cleanup_phase1.sh

# 2. Component Investigation (30 min)
# Manual: Check component usage with grep commands above
./cleanup_phase2.sh  # After investigation decisions

# 3. Documentation Updates (45 min)
# Manual: Update README.md, QUICK_START.md, GOOGLE_AI_INTEGRATION.md
git add -u
git commit -m "docs: Update to reflect Phases 2 & 3 completion"
```

---

## ⚠️ Safety Checks

Before executing cleanup:

1. **Backup current state:**
   ```bash
   git stash  # If uncommitted changes
   git branch backup-before-cleanup
   ```

2. **Verify no critical work in files to delete:**
   ```bash
   # Review each file before deleting
   cat IMPLEMENTATION_PLAN.md  # Check if has useful info
   cat IMAGE_GENERATION_USAGE.md
   ```

3. **Test after cleanup:**
   ```bash
   # Ensure tests still pass
   python3 test_gemini_integration.py
   python demo.py --quick
   ```

---

## 📝 Post-Cleanup Checklist

- [ ] .gitignore created and working
- [ ] config.json removed from git tracking
- [ ] config.template.json created
- [ ] All test outputs deleted
- [ ] All cache files deleted
- [ ] Outdated docs deleted or archived
- [ ] Component redundancy resolved
- [ ] Documentation updated to reflect current state
- [ ] Tests pass (10/10)
- [ ] Demo works (all 5 demos)
- [ ] Git commit with cleanup changes

---

## 🔮 Future Development Readiness

After cleanup, the codebase will be ready for:

### Phase 4: Advanced Features
- Multi-format diagram support (sequence, class, ER diagrams)
- Advanced layout optimization
- Template-based generation

### Phase 5: Production Enhancement
- REST API development
- Web interface
- Batch processing optimization
- Model fine-tuning with collected feedback

### Phase 6: Scale & Deploy
- Performance optimization
- Cloud deployment
- Monitoring & analytics
- User authentication & multi-tenancy

Clean foundation = faster future development! 🚀
