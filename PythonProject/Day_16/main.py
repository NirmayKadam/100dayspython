from prettytable import PrettyTable

table = PrettyTable()
table.add_column("Pokemon Name",["pikachu","squirtel"])
table.add_column("Type",["electric","water"])
table.align = "l"
print(table)
