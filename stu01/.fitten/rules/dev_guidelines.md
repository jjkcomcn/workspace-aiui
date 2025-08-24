# 开发规范指南

## 代码风格
1. **类型注解**:
   - 所有函数参数和返回值必须使用类型注解
   - 示例: `def func(param: int) -> str:`

2. **文档字符串**:
   - 公共函数/类必须包含完整的Google风格文档字符串
   - 包含: Args, Returns, Raises等部分
   - 示例:
     ```python
     """函数说明
     
     Args:
         param: 参数说明
         
     Returns:
         返回值说明
     """
     ```

## 错误处理
1. **参数验证**:
   - 关键函数开头必须验证参数类型
   - 使用raise TypeError/ValueError提示错误

2. **设备兼容性**:
   - 所有张量操作必须考虑设备兼容性(CPU/GPU)

## 模块设计
1. **单一职责**:
   - 每个类/函数只做一件事
   - 示例: ContentLoss只计算内容损失

2. **可扩展性**:
   - 关键参数应该设计为可配置
   - 示例: 损失权重和层配置

## 日志规范
1. **日志级别**:
   - DEBUG: 调试信息
   - INFO: 关键流程节点
   - WARNING: 非预期但可恢复的情况
   - ERROR: 需要干预的错误

2. **日志格式**:
   - 统一使用JSON格式
   - 包含: 时间戳、日志级别、模块名、消息体
   - 示例: 
     ```python
     import logging
     logging.basicConfig(format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":"%(message)s"}')
     ```

3. **关键日志点**:
   - 模型初始化
   - 损失计算
   - 权重更新
   - 异常捕获

## 测试要求
1. **单元测试**:
   - 所有核心功能必须有单元测试
   - 测试文件命名: test_*.py

2. **边界测试**:
   - 测试异常输入和边界条件
