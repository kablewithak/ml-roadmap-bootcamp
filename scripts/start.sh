#!/bin/bash
# ============================================================================
# ML Observability Stack - Startup Script
# ============================================================================
#
# This script handles the complete setup and startup of the observability stack.
#
# USAGE:
#   ./scripts/start.sh           # Start all services
#   ./scripts/start.sh --build   # Rebuild containers first
#   ./scripts/start.sh --stop    # Stop all services
#   ./scripts/start.sh --clean   # Stop and remove all data
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ML Observability Stack v2.0                          ║${NC}"
    echo -e "${BLUE}║  Production-Grade Observability for ML Services       ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
check_prereqs() {
    print_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker found: $(docker --version)"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose found: $(docker-compose --version)"

    # Check Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    print_success "Docker daemon is running"

    # Check available memory
    total_mem=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)
    if [ $total_mem -lt 8000000000 ]; then
        print_warning "Docker has less than 8GB RAM allocated. Recommend 8GB minimum."
        print_info "To increase: Docker Desktop → Settings → Resources → Memory"
    else
        print_success "Docker memory: $(($total_mem / 1024 / 1024 / 1024))GB"
    fi

    echo ""
}

# Start services
start_services() {
    print_info "Starting ML observability stack..."
    echo ""

    if [ "$1" == "--build" ]; then
        print_info "Building containers..."
        docker-compose build
    fi

    print_info "Starting all services..."
    docker-compose up -d

    echo ""
    print_info "Waiting for services to be healthy..."
    sleep 10

    # Check service health
    check_services
}

# Check service health
check_services() {
    echo ""
    print_info "Checking service health..."
    echo ""

    services=("fraud-api" "redis" "prometheus" "grafana" "jaeger" "loki")

    for service in "${services[@]}"; do
        if docker-compose ps $service | grep -q "Up"; then
            print_success "$service is running"
        else
            print_error "$service failed to start"
            print_info "Check logs: docker-compose logs $service"
        fi
    done

    echo ""
}

# Display URLs
show_urls() {
    echo ""
    print_header
    echo -e "${GREEN}🚀 All services are running!${NC}"
    echo ""
    echo "Access the following UIs:"
    echo ""
    echo -e "  ${BLUE}Fraud Detection API${NC}  http://localhost:8000"
    echo -e "  ${BLUE}API Documentation${NC}    http://localhost:8000/docs"
    echo -e "  ${BLUE}Grafana Dashboards${NC}   http://localhost:3000 (admin/admin)"
    echo -e "  ${BLUE}Prometheus${NC}            http://localhost:9090"
    echo -e "  ${BLUE}Jaeger Tracing${NC}        http://localhost:16686"
    echo -e "  ${BLUE}Alertmanager${NC}          http://localhost:9093"
    echo ""
    echo "Quick test:"
    echo ""
    echo "  # Populate cache"
    echo "  curl -X POST http://localhost:8000/debug/populate-cache/12345"
    echo ""
    echo "  # Make prediction"
    echo '  curl -X POST http://localhost:8000/v1/predict \\'
    echo '    -H "Content-Type: application/json" \\'
    echo '    -d '"'"'{"transaction_id": "test_001", "user_id": 12345, "amount": 99.99, "payment_method": "card"}'"'"''
    echo ""
    echo "Load testing:"
    echo "  pip install locust"
    echo "  locust -f chaos/load_test.py --host=http://localhost:8000"
    echo ""
    echo -e "${YELLOW}📚 For detailed documentation, see README.md${NC}"
    echo ""
}

# Stop services
stop_services() {
    print_info "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Clean everything
clean_all() {
    print_warning "This will stop all services and DELETE all data (metrics, logs, traces)"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Stopping services and removing volumes..."
        docker-compose down -v
        print_success "All data cleaned"
    else
        print_info "Cancelled"
    fi
}

# Main
main() {
    print_header

    case "$1" in
        --stop)
            stop_services
            ;;
        --clean)
            clean_all
            ;;
        --build)
            check_prereqs
            start_services "--build"
            show_urls
            ;;
        *)
            check_prereqs
            start_services
            show_urls
            ;;
    esac
}

main "$@"
