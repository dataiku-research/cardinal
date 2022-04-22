test-code:
	python -m pytest --pyargs cardinal

test-doc:
	pytest --doctest-glob='*.rst' `find doc/ -name '*.rst'`

dist:
	python setup.py sdist bdist_wheel

test-release:
	python -m twine upload --repository testpypi dist/*

release:
	python -m twine upload dist/*

test: test-code test-doc
