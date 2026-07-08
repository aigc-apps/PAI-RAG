# TODO

## 安全加固（后续版本，非当前版本范围）

以下为 JWT + RBAC 认证体系上线后已知、但**不在当前版本解决**的安全风险，留待后续迭代。相关实现见 `newbackend/app/auth.py`、`newbackend/app/routes/auth.py`。

- [ ] **登录限速 / 锁定**：`/v1/auth/login` 目前可被暴力尝试。生产必须加 IP / email 维度的 rate limit（及多次失败后的临时锁定）。
- [ ] **显式 CSRF 防护**：cookie 为 `SameSite=Lax`，能挡多数跨站 POST，但建议对所有状态变更接口加 `Origin` 校验或 CSRF token。
- [ ] **首次 admin bootstrap 部署风险**：若未设置 `ADMIN_BOOTSTRAP_TOKEN`，首次暴露服务时任何人都可能创建首个 admin。**生产必须设置** `ADMIN_BOOTSTRAP_TOKEN`。
- [ ] **`COOKIE_SECURE` 默认 false**：生产走 HTTPS 后必须设 `COOKIE_SECURE=true`。
- [ ] **JWT 无单 token 撤销**：禁用用户可通过每请求 status 检查即时失效，但没有 session 列表、踢单个设备、或 refresh token 轮换机制。
- [ ] **邀请链接 token 在 URL query**：可能进入浏览器历史、代理日志、Referer。短 TTL 可缓解；生产建议加一次性确认页或更严格的日志策略。
- [ ] **无 MFA / SSO**：admin 权限较高，建议后续接入企业 SSO / OIDC，或至少启用 MFA。
- [ ] **无审计日志**：邀请、禁用用户、Aliyun 授权、配置修改等操作需要审计记录。
