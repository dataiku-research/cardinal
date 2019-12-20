test-code:
	python -m pytest --pyargs cardinAL

test-doc:
	pytest --doctest-glob='*.rst' `find doc/ -name '*.rst'`

test: test-code test-doc
