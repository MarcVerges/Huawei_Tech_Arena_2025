setup:
	uv sync

lint:
	black src

test:
	flake8 src


run:
	@if [ -z "$(m)" ]; then \
		echo "Error: debes indicar el módulo a ejecutar:"; \
		echo "   make run m=main"; \
	else \
		.venv/bin/python -m src.$(m); \
	fi