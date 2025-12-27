# STA410 Quick Start Guide

## 5-Minute Setup

### Step 1: Clone and Install
```bash
git clone <repository>
cd STA410_Package
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
cp .env.example .env
# Edit .env with your Alpaca API credentials
nano .env
```

### Step 3: Run Tests
```bash
# Quick test to verify setup
pytest tests/ -v -k "test_valid" --tb=short

# Full test suite
pytest tests/
```

### Step 4: Use the System
```python
from stock_prediction.utils.config import get_config
from stock_prediction.utils.logger import setup_logger
from stock_prediction.core import StockPredictor

# Setup
config = get_config()
logger = setup_logger(__name__)

# Use system
logger.info(f"Environment: {config.environment}")
predictor = StockPredictor("AAPL", start_date="2023-01-01")
```

---

## Docker Quick Start

### Step 1: Build
```bash
docker build -t sta410:latest .
```

### Step 2: Run
```bash
docker-compose up -d
docker-compose logs -f sta410
```

### Step 3: Test
```bash
docker-compose exec sta410 pytest tests/ -v
```

---

## Common Commands

### Testing
```bash
# Run specific test
pytest tests/test_validation.py::TestOHLCVValidation::test_valid_ohlcv_data -v

# Run with coverage
pytest tests/ --cov=stock_prediction --cov-report=html

# Run unit tests only (fast)
pytest tests/ -m unit -v
```

### Configuration
```bash
# View current config
python -c "from stock_prediction.utils.config import get_config; print(get_config().to_dict())"

# Check logging level
python -c "from stock_prediction.utils.logger import setup_logger; logger = setup_logger('test'); logger.debug('DEBUG')"
```

### Docker
```bash
# View logs
docker-compose logs sta410

# Execute Python
docker-compose exec sta410 python -c "print('Hello')"

# Restart services
docker-compose restart

# Remove everything
docker-compose down -v
```

---

## Troubleshooting

### "Missing API credentials"
```bash
# Check environment
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY

# Or check .env file
cat .env | grep ALPACA
```

### "Configuration validation failed"
```bash
# Check .env syntax
# Values must not have spaces around =
CORRECT=value
WRONG = value
```

### "Data validation error"
```python
from stock_prediction.utils.validation import DataValidator
from stock_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    DataValidator.validate_ohlcv_data(df, symbol="AAPL")
except Exception as e:
    logger.error(f"Validation failed: {e}")
    logger.debug(f"Data shape: {df.shape}")
    logger.debug(f"Data columns: {df.columns.tolist()}")
```

### "Tests fail on import"
```bash
# Ensure package is installed
pip install -e .

# Check Python path
python -c "import stock_prediction; print(stock_prediction.__file__)"
```

---

## Environment Variables (Most Important)

```bash
# Security (CRITICAL)
ALPACA_API_KEY=pk_...
ALPACA_SECRET_KEY=...

# Operations
ENVIRONMENT=development      # or staging, production
LOG_LEVEL=INFO              # or DEBUG, WARNING, ERROR
PAPER_TRADING=true          # MUST BE true except with careful testing

# Configuration
TRAIN_TEST_SPLIT=0.8
RANDOM_SEED=42
MAX_PORTFOLIO_RISK=0.05
```

---

## Next Steps

1. **Read**: `PRODUCTION_GUIDE.md` for detailed documentation
2. **Review**: `.env.example` for all configuration options
3. **Test**: Run `pytest tests/ -v` to verify setup
4. **Explore**: Review modules in `stock_prediction/utils/`
5. **Deploy**: Use `docker-compose up -d` for production

---

## Support

- **Logs**: Check `logs/` directory for debug information
- **Tests**: Run `pytest tests/ -vv` for detailed output
- **Config**: Review `PRODUCTION_GUIDE.md` Configuration section
- **Docker**: Run `docker-compose logs -f` for container output

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `.env` | Your local configuration (don't commit) |
| `.env.example` | Template for .env |
| `stock_prediction/utils/logger.py` | Logging setup |
| `stock_prediction/utils/config.py` | Configuration |
| `stock_prediction/utils/validation.py` | Data validation |
| `tests/` | All test files |
| `PRODUCTION_GUIDE.md` | Complete documentation |

---

Good luck! 🚀
