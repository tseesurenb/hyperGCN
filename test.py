import polars as pl

 
df = pl.DataFrame(
    {
        "a": [1, 2, 3, 1],
        "b": ["a", "b", "c", "c"],
    }
)


new_df = df.with_columns(
    (pl.col("a") * 2).alias("a_times_2"),
)

print(new_df)