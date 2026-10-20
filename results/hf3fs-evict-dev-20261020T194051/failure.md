# 失败记录：http 探针的期望写错了（2026-10-20 19:40）

单元测试是 `25 passed`，随后 harness 的 `http_probe()` 在下面这一行抛
`AssertionError`：

```python
assert dict(client.allocate_pages_for_keys(0, [('third', '')])) == {'third': -1}
```

当时的探针只给 `cached` 加了租约，而池里另一个键 `new` **没有**租约。淘汰逻辑
因此合法地选中 `new`，`third` 成功拿到页 1（服务端日志：
`Evicted key new, returned 1 pages`）。

结论：**这是探针期望错了，不是实现错了**。要证明“租约能挡住淘汰”，必须让池里
所有 complete key 都被租用；释放其中一个之后，淘汰才应落在被释放的那个键上。

修复见 `results/hf3fs-evict-dev-20261020T194325/`：探针先 `acquire_keys(['cached','new'])`，
断言 `third == -1`，再 `release_keys(['cached'])`，断言 `third` 拿到 `cached` 的旧页、
`new` 仍然存在。

此目录没有 `result.json`，因为 harness 在写报告之前就退出了；`pytest.log` 与源文件
快照仍保留，用于对照修复前后的差异。
