function countrecord(df)
    sort(combine(groupby(df, All()), nrow => :__count), :__count)
end