# Contribution Guidelines

## Pull requests are always welcome

We're trying very hard to keep our systems simple, lean and focused. We don't
want them to be everything for everybody. This means that we might decide
against incorporating a new request.

## Create issues...

Any significant change should be documented as a GitHub issue before anybody
starts working on it.

### ...but check for existing issues first!

Please take a moment to check that an issue doesn't already exist documenting
your request. If it does, it never hurts to add a quick "+1" or "I need this
too". This will help prioritize the most common requests.

## Conventions

Fork the repository and make changes on your fork on a branch:

1. Create the right type of issue (defect, enhancement, test, etc)
2. Name the branch N-something where N is the number of the issue.

Note that the maintainers work on branches in this repository.

Work hard to ensure your pull request is valid. This includes code quality,
clear naming, and including unit tests. Please read the Code Of Conduct at the
bottom of this file.

Pull request descriptions should be as clear as possible and include a reference
to all the issues that they address. In GitHub, you can reference an issue by
adding a line to your commit description that follows the format:

`Fixes #N`

where N is the issue number.

## Merge approval

Repository maintainers will review the pull request and make sure it provides
the appropriate level of code quality & correctness.

## How are decisions made?

Short answer: with pull requests to this repository.

All decisions, big and small, follow the same 3 steps:

1. Open a pull request. Anyone can do this.

2. Discuss the pull request. Anyone can do this.

3. Accept or refuse a pull request. The relevant maintainers do this (see below
   "Who decides what?")

   1. Accepting pull requests

      1. If the pull request appears to be ready to merge, approve it.

      2. If the pull request has some small problems that need to be changed,
         make a comment addressing the issues.

      3. If the changes needed to a PR are small, you can add a "LGTM once the
         following comments are addressed..." this will reduce needless back and
         forth.

      4. If the PR only needs a few changes before being merged, any MAINTAINER
         can make a replacement PR that incorporates the existing commits and
         fixes the problems before a fast track merge.

   2. Closing pull requests

      1. If a PR appears to be abandoned, after having attempted to contact the
         original contributor, then a replacement PR may be made. Once the
         replacement PR is made, any contributor may close the original one.

      2. If you are not sure if the pull request implements a good feature or
         you do not understand the purpose of the PR, ask the contributor to
         provide more documentation. If the contributor is not able to
         adequately explain the purpose of the PR, the PR may be closed by any
         MAINTAINER.

      3. If a MAINTAINER feels that the pull request is sufficiently
         architecturally flawed, or if the pull request needs significantly more
         design discussion before being considered, the MAINTAINER should close
         the pull request with a short explanation of what discussion still
         needs to be had. It is important not to leave such pull requests open,
         as this will waste both the MAINTAINER's time and the contributor's
         time. It is not good to string a contributor on for weeks or months,
         having them make many changes to a PR that will eventually be rejected.

## Who decides what?

All decisions are pull requests, and the relevant maintainers make decisions by
accepting or refusing pull requests. Review and acceptance by anyone is denoted
by adding a comment in the pull request: `LGTM`. However, only currently listed
`MAINTAINERS` are counted towards the required majority.

The maintainers will be listed in the MAINTAINER file, all these people will be
in the employment of Pfizer.

## I'm a maintainer, should I make pull requests too?

Yes. Nobody should ever push to master directly. All changes should be made
through a pull request.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting one of the maintainers. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
