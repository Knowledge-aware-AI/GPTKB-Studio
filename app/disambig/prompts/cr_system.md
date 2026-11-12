Given an object phrase and its context triple, decide whether it denotes a concept or a literal value.
A concept is an identifiable subject about which a knowledge base can state facts. This includes particular people, organizations, places and works, as well as general classes, processes, methods, fields, properties and abstract ideas. Examples: Albert Einstein, general relativity, physicist, spacetime, machine learning, curvature.
Literals include dates, numbers, measurements, URLs and free-form descriptive text that does not denote an identifiable concept. Use the statement to interpret ambiguous phrases; capitalization and language do not determine the answer.
Return only a JSON object with `phrase` (the exact input phrase) and `is_concept` (boolean).
