# 🚀 MLPulse — API de Machine Learning com Stack DevOps Completa

![Build Status](https://github.com/tinalmeid/mlpulse/actions/workflows/mlpulse-ci.yml/badge.svg)
![Coverage](https://sonarcloud.io/api/project_badges/measure?project=tinalmeid_mlpulse&metric=coverage)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=tinalmeid_mlpulse&metric=alert_status)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Desenvolvimento

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pytest](https://img.shields.io/badge/Testes-Pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)
![VS Code](https://img.shields.io/badge/IDE-VS_Code-007ACC?style=flat&logo=visualstudiocode&logoColor=white)

## Infraestrutura & DevOps

![AWS](https://img.shields.io/badge/Cloud-AWS-FF9900?style=flat&logo=amazonaws&logoColor=white)
![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC?style=flat&logo=terraform&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white)
![Prometheus](https://img.shields.io/badge/Métricas-Prometheus-E6522C?style=flat&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Dashboard-Grafana-F46800?style=flat&logo=grafana&logoColor=white)

## Gestão & Qualidade

![Jira](https://img.shields.io/badge/Gestão-Jira-0052CC?style=flat&logo=jira&logoColor=white)
![SonarCloud](https://img.shields.io/badge/Quality-SonarCloud-F3702A?style=flat&logo=sonarcloud&logoColor=white)
![WakaTime](https://img.shields.io/badge/Produtividade-Wakatime-000000?style=flat&logo=wakatime&logoColor=white)
![Clean Code](https://img.shields.io/badge/Prática-Clean_Code-green?style=flat)

API de Machine Learning que recebe dados, treina modelos e serve predições via HTTP.
Projeto prático da trilha **SRE/DevOps Sênior** — evolui do localhost até produção na AWS
com observabilidade completa, IaC e pipeline CI/CD.

---

## 🚀 Quick Start

### Pré-requisitos

- Python 3.12+
- Docker + Docker Compose
- WSL2 (se Windows)

### Instalação

1. **Clone o repositório:**
```bash
   git clone https://github.com/tinalmeid/mlpulse
   cd mlpulse
```

2. **Crie o ambiente virtual:**
```bash
   python -m venv .venv
   source .venv/bin/activate
```

3. **Instale as dependências:**
```bash
   pip install -r requirements.txt
```

4. **Rode a API:**
```bash
   uvicorn app.main:app --reload
```

5. **Acesse a documentação:**
   - API: http://localhost:8000
   - Docs interativos: http://localhost:8000/docs

---

## 📡 Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status da API e do modelo |
| GET | `/metrics` | Métricas para o Prometheus |
| POST | `/train` | Treina o modelo com dados enviados |
| POST | `/predict` | Retorna predições do modelo |
| GET | `/history` | Histórico de predições (banco RDS) |

### Exemplo de uso
```bash
# Treinar
curl -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"x": [1,2,3,4,5], "y": [2,4,6,8,10]}'

# Predizer
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"x": [6, 7, 8]}'
```

---

## 🧪 Testes
```bash
pytest tests/ -v
```

---

## 🏗️ Estrutura do Projeto
```text
mlpulse/
├── .github/
│   └── workflows/
│       └── mlpulse-ci.yml       # Pipeline CI/CD
├── app/
│   └── main.py                  # API FastAPI
├── models/                      # Modelos treinados (gerados em runtime)
├── terraform/
│   ├── modules/                 # Módulos reutilizáveis
│   └── environments/            # Dev e Prod
├── tests/
│   └── test_api.py              # Testes da API
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

| Fase | Descrição | Status |
|------|-----------|--------|
| **S0** | Ambiente + API local + testes | 🚧 Em andamento |
| **M1** | Deploy AWS (EC2, S3, RDS) | 📅 Planejado |
| **M3** | IaC Terraform + CI/CD | 📅 Planejado |
| **M2** | Observabilidade (Prometheus + Grafana) | 📅 Planejado |
| **M4** | SLOs + Runbooks + Portfolio | 📅 Planejado |

---

## 📝 Padrões de Desenvolvimento

### Branching Strategy
- `MLP-XXX-tipo/descricao` — Ex: `MLP-001-feat/api-health-endpoint`

### Conventional Commits
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `chore`: Configuração e manutenção
- `docs`: Documentação
- `test`: Testes
- `refactor`: Melhoria sem alterar funcionalidade

### Quality Gate
- Cobertura mínima: 80%
- Zero bugs e vulnerabilidades no SonarCloud
- Nenhum código entra na main sem passar no CI

---

👩🏽‍💻 Desenvolvido por **Cristina de Almeida** como projeto prático da trilha SRE/DevOps Sênior.
