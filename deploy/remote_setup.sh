#!/usr/bin/env bash
# ============================================================
# 云服务器初始化脚本（Ubuntu）—— 幂等，可重复执行
#
# 用法（从本机执行，脚本本身不留在服务器上）：
#   ssh -i ~/.ssh/tencent_rag ubuntu@<IP> 'bash -s' < deploy/remote_setup.sh
#
# 做的事：装 Docker + compose plugin → 加当前用户进 docker 组 → 建部署目录
# 不做的事：不碰 ufw（防火墙规则靠腾讯云安全组，启用 ufw 若顺序不对会把自己挡在门外）
# ============================================================

set -euo pipefail

DEPLOY_DIR=/opt/rag-zhiku

echo "==> 环境检查"
. /etc/os-release
echo "    系统: ${PRETTY_NAME:-unknown}   架构: $(uname -m)"
if [ "${ID:-}" != "ubuntu" ] && [ "${ID_LIKE:-}" != "ubuntu" ]; then
  echo "    !! 非 Ubuntu 系，脚本未验证，请先确认再继续"
fi

echo "==> 安装基础依赖"
sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl gnupg

echo "==> 安装 Docker（已装则跳过）"
if command -v docker >/dev/null 2>&1; then
  echo "    已安装: $(docker --version)"
else
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  echo "    安装完成: $(docker --version)"
fi

echo "==> 配置镜像加速（腾讯云内网源，非腾讯云 CVM 上会静默无效，无害）"
if [ ! -f /etc/docker/daemon.json ]; then
  sudo mkdir -p /etc/docker
  sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "registry-mirrors": ["https://mirror.ccs.tencentyun.com"]
}
JSON
  sudo systemctl restart docker
  echo "    已写入 /etc/docker/daemon.json"
else
  echo "    /etc/docker/daemon.json 已存在，未改动"
fi

echo "==> 将当前用户加入 docker 组（下次登录生效）"
if id -nG "$USER" | grep -qw docker; then
  echo "    $USER 已在 docker 组"
else
  sudo usermod -aG docker "$USER"
  echo "    已加入，需重新建立 SSH 连接后才生效；本脚本后续仍用 sudo docker"
fi

echo "==> 创建部署目录 ${DEPLOY_DIR}"
sudo mkdir -p "${DEPLOY_DIR}"
sudo chown "$USER":"$USER" "${DEPLOY_DIR}"
ls -ld "${DEPLOY_DIR}"

echo "==> 预建数据目录并修正属主"
# Milvus 容器以 uid 999(milvus) 运行；若目录由 Docker 隐式创建则为 root:root，Milvus 会 panic: failed to mkdir
# 先建好并改属主，Docker 便会沿用已有目录
mkdir -p "${DEPLOY_DIR}/volumes/milvus"
sudo chown -R 999:999 "${DEPLOY_DIR}/volumes/milvus"
ls -ld "${DEPLOY_DIR}/volumes/milvus"

echo
echo "==> 完成。下一步："
echo "    1) 从本机上传部署产物："
echo "       scp -i ~/.ssh/tencent_rag -r deploy/docker-compose.yml deploy/user.yaml deploy/mongo-init.js deploy/.env ubuntu@<IP>:${DEPLOY_DIR}/"
echo "    2) 在服务器上启动：cd ${DEPLOY_DIR} && sudo docker compose up -d"
