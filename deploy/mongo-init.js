// MongoDB 首次初始化脚本（容器 entrypoint 在初始化阶段以 root 身份执行一次）
// 作用：建一个只对业务库有 readWrite 权限的应用账号，避免应用直接用 root 跑
// 密码通过容器环境变量传入，不写死在文件里
db.getSiblingDB(process.env.MONGO_INITDB_DATABASE).createUser({
  user: process.env.MONGO_APP_USER,
  pwd: process.env.MONGO_APP_PASSWORD,
  roles: [
    { role: 'readWrite', db: process.env.MONGO_INITDB_DATABASE }
  ]
});

print(`[init] 应用账号 ${process.env.MONGO_APP_USER} 已创建于库 ${process.env.MONGO_INITDB_DATABASE}`);
