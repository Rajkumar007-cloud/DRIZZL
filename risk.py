def calculate_risk_score(corrected,confidence,reliability):

    score = min(100,int((corrected /250)*50 +confidence * 0.25 + reliability * 0.25))

    return score


def get_risk_level(score):
    if score < 25:
        return "🟢 Low"
    elif score < 50:
        return "🟡 Moderate"
    elif score < 75:
        return "🟠 High"
    else:
        return "🔴 Extreme"


def get_risk_message(score):
    if score >= 75:
        return """
🔴 EXTREME RAINFALL ALERT

• Urban flooding possible
• Waterlogging likely
• Travel disruption possible
• Emergency preparedness advised
"""

    elif score >= 50:
        return """
🟠 HIGH RAINFALL RISK

• Local waterlogging possible
• Reduced visibility
• Traffic delays possible
"""

    elif score >= 25:
        return """
🟡 MODERATE RAINFALL RISK

• Intermittent heavy showers
• Minor inconvenience possible
"""

    else:
        return """
🟢 LOW RAINFALL RISK

No major rainfall-related impacts expected.
"""