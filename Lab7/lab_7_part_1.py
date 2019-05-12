import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Temperature ranges [0, 45]
#   * Power has a range of [0, 100]

x_temp = np.arange(0, 45, 1)
x_power = np.arange(0, 100, 1)

# Generate fuzzy membership functions
temp_cold = fuzz.trimf(x_temp, [0, 8, 23])
temp_pleasant = fuzz.trimf(x_temp, [6, 20, 30])
temp_hot = fuzz.trimf(x_temp, [23, 35, 45])

power_low = fuzz.trimf(x_power, [0, 15, 30])
power_mid = fuzz.trimf(x_power, [30, 45, 60])
power_high = fuzz.trimf(x_power, [60, 75, 100])

# This is for visualization purposes.
power0 = np.zeros_like(x_power)

# Visualize these universes and membership functions
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 6))

ax0.plot(x_temp, temp_cold, 'b', linewidth=1.5, label='Cold')
ax0.plot(x_temp, temp_pleasant, 'g', linewidth=1.5, label='Pleasant')
ax0.plot(x_temp, temp_hot, 'r', linewidth=1.5, label='Hot')
ax0.set_title('Temperature')
ax0.legend()

ax1.plot(x_power, power_low, 'b', linewidth=1.5, label='Low')
ax1.plot(x_power, power_mid, 'g', linewidth=1.5, label='Mid')
ax1.plot(x_power, power_high, 'r', linewidth=1.5, label='High')
ax1.set_title('Power')
ax1.legend()

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()


def apply_rule_to_get_activation_values(temp, visualizing=False):
    temp_level_cold = fuzz.interp_membership(x_temp, temp_cold, temp)
    temp_level_pleasant = fuzz.interp_membership(x_temp, temp_pleasant, temp)
    temp_level_hot = fuzz.interp_membership(x_temp, temp_hot, temp)

    # Now we apply this by clipping the top off the corresponding output
    # membership function with `np.fmin`
    power_activation_low = np.fmin(temp_level_cold, power_low)
    power_activation_mid = np.fmin(temp_level_pleasant, power_mid)
    power_activation_high = np.fmin(temp_level_hot, power_high)

    # Visualize this
    if visualizing:
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.fill_between(x_power, power0, power_activation_low, facecolor='b', alpha=0.7)
        ax0.plot(x_power, power_low, 'b', linewidth=0.5, linestyle='--', )
        ax0.fill_between(x_power, power0, power_activation_mid, facecolor='g', alpha=0.7)
        ax0.plot(x_power, power_mid, 'g', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_power, power0, power_activation_high, facecolor='r', alpha=0.7)
        ax0.plot(x_power, power_high, 'r', linewidth=0.5, linestyle='--')
        ax0.set_title('Output membership activity')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()

    return power_activation_low, power_activation_mid, power_activation_high


fuzzifications = apply_rule_to_get_activation_values(30, True)


def defuzzify(fuzzifications, visualizing=False):
    # Aggregate all three output membership functions together
    aggregated = np.fmax(fuzzifications[0],
                         np.fmax(fuzzifications[1], fuzzifications[2]))

    # Calculate defuzzified result
    power = fuzz.defuzz(x_power, aggregated, 'centroid')
    power_activation = fuzz.interp_membership(x_power, aggregated, power)  # for plot

    # Visualize this
    if visualizing:
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.plot(x_power, power_low, 'b', linewidth=0.5, linestyle='--', )
        ax0.plot(x_power, power_mid, 'g', linewidth=0.5, linestyle='--')
        ax0.plot(x_power, power_high, 'r', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_power, power0, aggregated, facecolor='Orange', alpha=0.7)
        ax0.plot([power, power], [0, power_activation], 'k', linewidth=1.5, alpha=0.9)
        ax0.set_title('Aggregated membership and result (line)')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()


defuzzify(fuzzifications, True)
