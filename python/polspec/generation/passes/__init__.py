"""The passes that rewrite a generated frame, one module per kind.

The engine fills every column on its own; these then make the frame satisfy
what spans columns or rows -- a rule's condition, a foreign key's parent, a
hierarchy's shape, a composite key's distinctness. Each is a function of the
frame, the declaration and a seed, and `polspec.pass_order` decides the order
they run in. The declarations they read (`ColRule`, `ForeignKey`,
`Hierarchy`) live beside `ColSpec`, and know nothing of these.
"""
