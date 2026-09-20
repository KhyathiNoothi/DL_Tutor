from loss import mean_squared_error


print("===== LOSS CALCULATOR =====")

actual = float(input("Enter actual value: "))
predicted = float(input("Enter predicted value: "))

loss = mean_squared_error(actual, predicted)

print("\n===== RESULT =====")
print("Actual    :", actual)
print("Predicted :", predicted)
print("MSE Loss  :", loss)