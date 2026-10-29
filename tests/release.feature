Feature: Reduced article release

  Scenario: Inspect code and reported results
    Given the research pipeline and prompt templates
    And frozen result tables and figures
    When the release integrity checks run
    Then all released files match the manifest
    And withheld inputs and intermediate artifacts are absent

  Scenario: Explain the limits of reproduction
    Given evaluation inputs are withheld for future testing
    When the offline reproduction entry point runs without those inputs
    Then it exits with an explanation of the missing private data
