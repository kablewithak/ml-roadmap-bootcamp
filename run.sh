#!/bin/bash

# Fraud Detection System Quick Start Script

set -e

echo "🚀 Starting Fraud Detection System..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start infrastructure
echo "📦 Starting infrastructure (Redis, Kafka, Zookeeper, PostgreSQL)..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check Redis
echo "🔍 Checking Redis..."
until docker exec fraud-redis redis-cli ping > /dev/null 2>&1; do
    echo "  Waiting for Redis..."
    sleep 2
done
echo "  ✅ Redis is ready"

# Check Kafka
echo "🔍 Checking Kafka..."
until docker exec fraud-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; do
    echo "  Waiting for Kafka..."
    sleep 2
done
echo "  ✅ Kafka is ready"

# Check PostgreSQL
echo "🔍 Checking PostgreSQL..."
until docker exec fraud-postgres pg_isready -U payment_user > /dev/null 2>&1; do
    echo "  Waiting for PostgreSQL..."
    sleep 2
done
echo "  ✅ PostgreSQL is ready"

echo ""
echo "✅ All infrastructure services are ready!"
echo ""
echo "📝 Next steps:"
echo "  1. Install Python dependencies: pip install -r requirements.txt"
echo "  2. Configure environment: cp .env.example .env (edit with your Stripe keys)"
echo "  3. Start API server: uvicorn src.api.main:app --reload --port 8000"
echo ""
echo "📊 Service URLs:"
echo "  API:          http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Redis:        localhost:6379"
echo "  Kafka:        localhost:9092"
echo "  PostgreSQL:   localhost:5432"
echo ""
echo "🧪 Run tests:"
echo "  pytest tests/ -v"
echo ""
echo "📈 Run benchmarks:"
echo "  python benchmarks/fraud_detector_benchmark.py"
echo ""
