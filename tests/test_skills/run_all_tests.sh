#!/bin/bash
# 运行所有 skill 模块测试并生成报告

echo "=========================================="
echo "Skill 模块测试汇总报告"
echo "=========================================="
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 创建报告文件
REPORT_FILE="/tmp/skill_test_report.txt"
echo "Skill 模块测试汇总报告" > "$REPORT_FILE"
echo "生成时间: $(date)" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 定义模块列表
MODULES=(
    "schema"
    "config"
    "loader"
    "builder"
    "store"
    "retrieval"
    "execution"
    "embedding"
    "initializer"
    "market"
    "gateway"
)

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_ERRORS=0

for module in "${MODULES[@]}"; do
    echo "测试模块: $module"
    echo "----------------------------------------"
    
    if [ -d "tests/test_skills/$module" ]; then
        # 运行测试并捕获结果
        OUTPUT=$(python -m pytest "tests/test_skills/$module" -v --tb=no 2>&1)
        
        # 提取结果
        PASSED=$(echo "$OUTPUT" | grep -oP '\d+(?= passed)' || echo "0")
        FAILED=$(echo "$OUTPUT" | grep -oP '\d+(?= failed)' || echo "0")
        ERRORS=$(echo "$OUTPUT" | grep -oP '\d+(?= error)' || echo "0")
        
        PASSED=${PASSED:-0}
        FAILED=${FAILED:-0}
        ERRORS=${ERRORS:-0}
        
        TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
        TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
        TOTAL_ERRORS=$((TOTAL_ERRORS + ERRORS))
        
        echo "  ✓ Passed: $PASSED"
        echo "  ✗ Failed: $FAILED"
        echo "  ⚠ Errors: $ERRORS"
        echo ""
        
        # 写入报告
        echo "模块: $module" >> "$REPORT_FILE"
        echo "  Passed: $PASSED" >> "$REPORT_FILE"
        echo "  Failed: $FAILED" >> "$REPORT_FILE"
        echo "  Errors: $ERRORS" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    else
        echo "  目录不存在，跳过"
        echo ""
    fi
done

echo "========================================"
echo "测试汇总:"
echo "  Total Passed: $TOTAL_PASSED"
echo "  Total Failed: $TOTAL_FAILED"
echo "  Total Errors: $TOTAL_ERRORS"
echo "========================================"

echo "" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"
echo "总计:" >> "$REPORT_FILE"
echo "  Passed: $TOTAL_PASSED" >> "$REPORT_FILE"
echo "  Failed: $TOTAL_FAILED" >> "$REPORT_FILE"
echo "  Errors: $TOTAL_ERRORS" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

cat "$REPORT_FILE"
