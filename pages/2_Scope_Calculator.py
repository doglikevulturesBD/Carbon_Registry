import streamlit as st
from scopes.scope1 import calc_scope1
from scopes.scope2 import calc_scope2
from scopes.scope3 import calc_scope3

def main():
    st.title("📊 Scope 1 / 2 / 3 Calculator")

    choice = st.radio(
        "Select scope:",
        ["Scope 1 – Direct", "Scope 2 – Purchased Energy", "Scope 3 – Value Chain"],
        horizontal=True,
    )

    if choice.startswith("Scope 1"):
        calc_scope1()
    elif choice.startswith("Scope 2"):
        calc_scope2()
    else:
        calc_scope3()

if __name__ == "__main__":
    main()

