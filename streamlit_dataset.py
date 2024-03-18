import streamlit as st
from PIL import Image
import cv2 as cv
import numpy as np




st.title('1. Image from Path')
img = Image.open( "Screenshot (562).png")
st.image(img)
st.title('2. Image from Link')
st.image('data:image/jpeg;base64,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', width=700)
# st.title('3. Video')
# video_file = open('/path/to/your/video.mp4', 'rb')
# st.video(video_file, start_time=20)
# st.title('4. Audio')
# audio_file = open('/path/to/your/audio.mp3', 'rb')
# st.audio(audio_file.read())




st.title('1. Image from Path')
img = Image.open('Screenshot (563).png')
st.image(img)

st.title('2. Image from Link')
st.image('data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBUVEhgSFRQYFRgZGBIRGBoYGhkYEhgZGRgcGRgYGBgcIS4lHB4rIRgYJjgmKy8xNTY1GiQ7QDs0Py41NTEBDAwMEA8QHhISHjQrJCs0NDQ0NDQ0NDQ0NDQ0NjQ0NDQ0NDQ9NDQ0NDQ0NjQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NP/AABEIAKgBLAMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAAAAQIDBAUHBgj/xABJEAACAQIDBAUIBwUFBwUAAAABAgADEQQSIQUxQVEGE2FxkQciMlKBkqHRFGJysdLh8EKCorPBFSM0dJNDg4TCw+LxJDNEY7L/xAAZAQEBAQEBAQAAAAAAAAAAAAAAAQIDBAX/xAAlEQACAgICAgICAwEAAAAAAAAAAQIRAyESMQRRE0EiYTKBsRT/2gAMAwEAAhEDEQA/AOUwhCe8wEIQgBCEIAQhCAEIQgtBCEIFBCEJRQQjgkkVJaFEVouWThI4U4oaK2WGWWurgactC0VcsS0slI005CFeElKRhWSgNhCEAIQhIAhCEAIQhACbuG6M1HpoUdDVemMSlDzg7Ui+QPnIyBidcpPo63vpMKbWz+klekEUZGCGmoJRRWNNagq9R1tswQsN3bbdpMyv6A09GcXcjqb2F9KlNgTd1yKwezvenUGRbt5h00j06MYgVUpVQlHP1pNRnRqaCkgqVC5QtkyqyGxsfPXnJavS7FEuFZER1K5FRQi3Z3zKAB596tQ345t2gtA3SbFXdhUyM5qMXQZHBqdTnysNVuMPTGnAMP2jJ+X6GiOvsVkwz13qKrJWbCtSyv1gqLcsM1stgATv4W3zKmtjukFatTem4pkO4ruwRRUapaxfOP2iNCeNzzmTNK/sBCLlPKL1Z5HwmgNjkQsQo3k2/XZHCix/ZPhp4xWYAZV1vozc+wfV++KAlenkYpmVrG10OZD3HjI4Rb8oLQkIuY8zFznmfEwWhsLx4qN6zeJi9c/rt7x+ctFUSO8cUNg1jY3ANjlJG8A8bXHjJBXf1395vnJHxLui03dmVSzKrEkAtbMRfibDwg1SK4WSKkeiSwlOEjDZClOTrSk6Upap4ebUTjPKolNaUkFGa9PZT2BYBBvu5C3HMA6n2AyX6JSX0ql/spcfxFfulpHll5Xow+pgaM3MlDm590fDWIaVE7ncfuq3xzCaqJz/AOqXowmoyJqU9A2zlPoOjdhOU/xWHxlTEYJkNmUjvFvaOcnE3Hy0+zFenInSarUOJ3ff3SB17B4CYcaPVDKpdGYySMiX3vyX3V+UgZjyX3V+UlHXsrQkxc8l91flFp4gqQwVDYg6ohGhvqLbpKRNkEJPjMQalR6jKql2ZyEUIgJN7Ko3CQSCwhCECwhCEhQhCEAIQhACEIQAtCFotoLQkItjyiSlQRbRIolOiQ4Ac/hFAHP4fnEAj1WRs2oj1XtPh+cmSn2nw/OJTpzQwODLtYFVsCxZzlQAczwvuHaYsvAgp0r7j8N8t0sLfX9CTYbDXnodn7PFuse4A000Zza+Re3meAPMgHXJJHDLFpGbgdllhmJCIDYu26/IDezdg9thrLFXFpT0prY+u1jU9nBPZr2mG1NpMdNwGiqNFUcgP0TvNzPN4rGMeMnM8y8WU3b6LeJ2kSSSbk+MqU671Gy00eoRvCKzsO8KDOjdCfJwGRcRjgTms6UNVsN4NUjW59ThxvqB7PH9Kdm4AdQalOmV06qiuZl7ClMHL7bTjLNuoqzovHhHSRwfE0cRTGapQrIvNqbqo7yyiVUxvbO74byl7NdsprlL6Xem6r7Wy2HtIjOkPQbBY6n1tHJSdhnStRylHvuLKPNcHnv7ZFma/kqK8UPRxSlju2auE2nYZSbr6p1U+w7j274N0I2mrugwzNlJQspTq3HNGYi4Onb3G4lLG9HsdhxmrYWqqjUtkzoBzZ0uo9pnZZV7PNl8SMujWaglUXTRvUO79w/0PiZmVsMRe43aflIMJi92txN6m61hY+nuU+tyVu3kfYdN3ZSvs+fJTwyPOmiWIVQSSQoA3knQAdsr4iiysVYFSpKkHQgjQgzcr0jTJto+4ninMD63bw791N0y+cdXOovrlvrma+9uQ9pklE9uLOpL9ma2Eqcrd7Kp8CbiN+hVeFv9Sn+KPqpcknUnU33ntld0nM9Sb/RL9Brdn+rT/HE2hg6lCq1CsLOpAYBlaxKhgQykg6EcZH9Fb6o3HznRTru0ZgRFGDbgaf8Aq0R97xst+2V3S3cdx4GNtL+JwjUlRi6EVAzgI6ORlbKc+W4U/f7JW65ufwHyjRSCF5P17c/gPlE69ufwHygbIYXkv0hufwHyijEvz+C/KQWQgwljHYx61VqtRszuQWayrcgADRQANANwleQoQhCUBCLaJBUEW45f0/pEhCNIdccj4j5RwK8j7w/DI44HslNpkqlPVb3h+GToU9VvfH4JWU9n3/OTow5D4/OSjopUXaOTir++o/5JfpuCAqjKo1te5J9ZjYXPsFvG+fSceqPj85qYN19RfFvnKoklmUVZubIwgYjNoNW13ZR6THsFvadBJdsbTFsiqFUaKOIAN9/M7z2xrVMlG59Kp5x7EU2UdgLAm31Fnnnz1qi0aalndgigbyT9w7fbOckc8eb5Jb6KeNxOY6n2z1fk26MPXxaV61JxRpjrlLoyo73GQKSLMAbtp6o5z3nRnoRh8HT6+vkqVVXO1R7dVTsLnJm0AHrHXuGk19gdKcNjKlWnh2Z+qFMs+W1Ns5cDITq3oHW1tRYmcJT06O08qaqKPIeVPpg9C2Cw7FHZc1V1NnRT6KKf2WI1J3gWtvuOOETZ6YYovj8U7anr66/u03KKPdRROobD8n+CwVMYnGutRlAYmoQuGQ77BT6Z4Xa97aATaahFHK0kc56M9CsXjSGRerpHfVqAhLfUG9z3acyJ1vZWAwWxMMesxDDOczGoxOdwNeqoroD9kE7rk2nmOkvlWGtLApp6PXOLD/d0z97W+yZ4HC4PF7QxByipiKptnZjcKOGZz5qLvsNByENSn/LSObZ03EeV7DhrJhqzr6zFEv2gXJ8bTS2N5TsFXYI+fDsTYGqB1ZP21JC97WngNreTTHUKYqKExGl3Wlcuv2VYDOO7XsnjwupGoIJBB0II0II4GVY4SWjMpNHbul/QCjiVathwtGv6V10pVeNnUaAn1xrzvOUYd2puyMrK6syMGFmQqbMtr773E9x5J+kj5/7PqNdSrNQJOqldWpj6trkDhYjdYBPKvsrJXp4pNBVBp1LGwzqBlbvK6fuCXHKUJcX/AEebyIKcOSMJ1Dpn/bUC/duDHtGg8DzmPVwblWqAXVSuY3GhY6dp1mlst7EE2tx1BuDvHtBMdtHD2Nh6I9HuOoPeRaexbPjxnwnR5p0jWp5NSPO3gHcvJmH3D2nt06lLLqRdt4B3D6zDieQ9p7c6oL6nU7zzmZRo+nizctFF11uTqbnmT2yMrLbUidQJG1BuUxT9HqUl7KxUfofnEy8pY+iuf2T8JC6EGx0MlNdm07ISISUi/f8Af+cZ1Z/REUUYYkl6o9niPnDqG5D3l+clMhFCS9Q3Ie8vzh1DdnvL84oWRxbjl8fyjYqqSbAXJ0EFHummZd27tB5H58Yy0leyEZWD+aM2hC34r9YaDUf0iV8gIyFiLAnNYENxAtvG7WGEyKLF36j2j+sbBUx1uyAHZGxwtyPj+UppMkUHl+vGTU1PL9eMgS3I+I+Us0ivJveH4ZVQbLdGmeX68Zr4CgxIAXU2A7zu4ygtNAgazKxsVBZTcesRlFhy5901NiL/AHit6p6w/uedr4Ta6PF5E2kWtu1fOIFgq2QH6qAKviADPS+R/ZSs9bGsPRP0alfgcoao/eQyC/DzhxM8FtV51vyQW/stbb+txGbvzm3wtPNn1E34juNni/K30meriDgUYilSy9YBueoQGs3NUBGnrX5C1/yGnz8X9nCffWnO+khb6dir+l9JxV/9Vp0TyGenjPs4T760zKKjj0eu/o510n/xeL/zGL/mvOv+WHDPUweHpojO7YqmFVAWYnqau4Ccg6T/AOMxf+Zxn8159EdJekNLA4cV6wYgkU1VBdmcqWCi+g0Um5PCTI2nGiHOui/kqdrVMa2VdD1NM+eeyo40XuW/2hPYdJNr09k4ZVoYNimoGRcuHQ+tVcXIJ52uTxnM9t+UbG13DU3+jIpzKlPVjbd1jsPO7rBeYM9Z0S8pgqsmGxiAO5WktRBemxchQHp71uSBcXGu5RJKM3uW/wBGbRkbG8qmJWoTiUWtTY3tTAR6Y5Jc2YdjG/1uE9hidl7N2xTNWmw6ywBqU/MxCHgKiEaj7Q7jxlTpL5MqFa9TCkYepqclr4dj9kap+7p9UzmGN2fi9n1wXV6DgnJUQnK32HXRhpqviJUoy3F0zEm13tHrMF0MxeC2jhny9bTFemOtQGwUnK2dd6aE8xrvnp/K9b6HSvv+kJb/AE6l5m9EPKSXZaGNygmyrWWyqTuHWLuW/rDTmANZseVDZFSthVrJ53UFqjpa4dSBmYdqgE25FuNpLlzXMzJJwaicy2ew5j4/KbWIsaatvIunYOIJv3nwnlMLW87lx03eyeipPek3YUb/APQ/5hPdFnwM8HGZlYob/GZjrNHEmZ1RpZHr8dMr1JXeTuZXczkz6cFoj7tIKwtlbdwPFfy7IEiIbdvj+Uh0A0/rr8flGmkPXX+L5QNuR8R8ol15N7w/DADqh6yfxfKS/Qh1b1OupeaUGTMetfMbXRSuoG867pDmT1W94fhjqrU+rUIrh7vnLMpQrpkCgAG41vBGQQhCQo4EcvjHZ9LKLX0JvckcuwRg/WkUX5fAfKCig5e/xA/P9dz0qOd1vdX5SMJz0/XCIzX7uAgFmq1R8ua3mDKtsgsL34b98aKLeoPH/ujsG6o61HRamtwjXyuPrW1t8TK6oOJ+H5ygt4zCPRydZSVc6LWTzic1NiyhvNc21VtDrpK/WD1V8X/FGhV5nwHzjgq+sfd/ONgt4DFojh2w9OoBe6ualjpb17eIMXDqAMxsTewX+p7JXVV9Y+6PxSekF9Y+6PxSoSZcohnbmTqSfiSeAm7sqpZiiHQpUDH1/MbwHL9AYaOLZVvbS5Ohb5Ds9vK2psh7VF5ZgD3E2PwJnSPR87ynopbU3me68i22lVq2BY2LH6TTvxNgtRR2gKht9rlPE7VQgm/dMzZpqDE0jSYpV6ymtNhvDMwVT2i51HEXE45Y8k0d/DlcD2flZ6NPRxTYxVvRrlSxG5KtrEHkGADA8yw5Xp+TTpVRwNar16tkrCkC6DNkKFyCyjUqc53XOg0PDt+1qtBaWXEtTFNytI9ZYU2LblObTW3Gc06T+SjfVwLdvUu2n+7qH7m5+kJ5ozTjxkexmvt3oJgtpIcThKq03fM2enZ6NRjqc6A6Ncm5Fjc6g7ozy1i2AodmKp/yaonLMDj8Zs/EEKXw1QWzowsrDhnRtHXfZvAzqGwPKVhsSn0fHIlIsMjMwzYR77w2a+Tua4+tK4yi0+0iWjjgl7YR/wDWYb/M4X+ak6ntryVUKriphK3UKxBZSDUp2OuamcwI7iSOVpr7O2Ns7ZWS7L11RkpI9Sz4h2YhQEUDzVuRfKAOZ4zTzKtEoo+U/pBiMHUwr4eplv8ASc6kZqbgdVYOvG1zqCCLnXWavRXbSbVwj9dQUBXNGohs6McobMtxcaMO0Hid88v5bz/hP+K/6Mv+RT/CYj/Mn+VTnPivjUvst7o47TbzRx0E7x5MtqNiNnqHOZqTNhmJ1LBQChN9/msovxsZwSn6I7hO0+RnDMuBqVDuqV3KdqqiIT7ysPZOuZfic4LZzbbOzzh8bWoKDlSo6rp+wfOQe6VmtgaTmm4Ck+aPgyn7gZU6ZVA+1cSy6jrAntSmiN8VMmTzaJ36so9gDX+9Z6MdtI+V5iXOjKxbyhUMtYlheU6s6SO3jRK7mQsZI5kJnE98UDNpbhJMIiZw1TN1YIz5LdYR6qX0vI8x5fARGJO/hoOAEGqGvYsct7XNr2zWvpe2l7b40tbd4/rhFY8v/MFUAZm/dHPtP1fv+IFGGofWPiYCo3rHxMU1TyX3E+UTrT9X3E/DFkDr29dveMXr39dveMnw2PenmyhPOR6LXRT5rrlYjTRrbjwlSSwFouQ8okLQUcKZ5R4XLqwueC/1bs7OPdvitFgDi283uTvMSEJQEcIgWPy93iIALJkMjVO0eIkyUzzX3l+ctBss0jL+GbWZ6qVNiLEb5bpNOkTxeRG4mttZM4z+sA/tPpfxBpi7McJjMPUbRUxGHduwLVUn4Cb2FbPTKcRd17recPgD7DzmHj8NvBGhmZxPN4mThLizs/lawhfZblRfq3pVSPqhsrH2BifZOVdGOm2LwVkVutoj/ZVCSoH1G3p3C6/VnXehG20x+B6upZ3Vfo+IRv2rrlzEcVdbnvzDhOUdMuhVbAuzqrVMMSStQalBwWrb0SN2bcewm08eOtxkfXftHScHtnZu2KYo1UAqWNqdTzK6niaTg+d+6d28DdI8B0F2bs/Nia75wrXVsQV6tNfNAUAKz9pBN7WAnEFO4jsII3gjcQZoGrisbUSmXrYpwLIpZ6jKNxIuTlG67G3aZp4munSJZ0HpP5VWa9LArlG7rqi+d306Z3d7+7PG9HMLicZj6dRVqV2WtQqVXJJyqrqxLO2g0BsL9gE9r0a8lO6pjnsN/U02t/qVB9y+8ZqbZ6fYLA0/o2CppVZbqFp2XDIeOZx6R+ze/EiZTS1BWH7ZneW//wCJ/wAV/wBGXvIn/hMR/mT/ACqc5dt7b2IxlQVMQ+ci4RQMqIDa4ReG4am5Nhcm07B5Jtnmjs3rH066o+I10smVUU9xFPN3NElxx0wnbOTdEejFfHOqU1K0xYPVI/u0HEA/tPyUe2w1nbtr4+hsrZ4ygAU1FGihPnO9vNB5km7MeWYzM2r5Q8BhqeSgwrsBZUoACkOX95bIo7rnsnJ9u7dr46t1tZt1wiL/AO3TU8FHEmwux1NuQAFqWR7VIzKShGyHBgu5dyzMzM7NpcsxuxPeSTNvGsqqiG+gzG1r3ax+4LK2ysMB5zDzVGY/L2mw9srY6uWJJ3kknvM9sEfFm+cyri2XO2TNlv5ua2a3bbSUXaTVGlZz2zMmfSwxqJGw5EeIEiK93iPnHOO0RhXtEwelBkPMe8vziGmea+8vzhl7REK/WHxgouUDU2PIAg37yDoPvkTMSbnUx5Uesvj+UAg3lgR2Xue7SANWncXJAG65vv5CwMMg9df4/wAMHe/3AcAOQkcMlEmQeuv8f4YZB66/x/hkcJALpF07fhGwgpIqqdASDwva3cTwjMtjY6cO2JJ0Oayk2O5Sdx5Kx+4+Om52QihJK9B0Yo6MjC11YFWFxcXB13EH2xiylFEWLp2xwt2+P5QBsesAV5HxHyjwU5N4j5S0Qu0q6mmVdSzjKEa9sgv5ysP2hbdyvHU2kFNk4q/sZfwyxVen5mQOPN8/OVN2ufRsBpa0J7OUo2qov4PEFSCDYggiaOMoB16xe4j1Ty7uX5TCpvNPBYsqefAg7iORnXtHy82NxlaINnY+vg6wr0GysLqQRdHXirLxX/yLGdV2D5SsJXAWufoz2sQ+tE87VLWA+1l9s55icKrrnTUcR+0vfzHb926YmJwRHCcMmJS7PTg8ulxkdvqbL2RV/vTTwT384sOqse0kb/bNjYtPCimfoi0QlyD1ATJmG8EppcT5peh2fCa/RvpFXwNQvRYZWtnRrmm4HMcG5MNe8aThLC67PfHLF9HpPKNiNqs7JiEZcNc5RQDNhmXgajDUnsewuNBxngKPnkKnnsdAq+cx7gNZ3DZPlRwVRQK2fDvoCGVnS/1XQHTtYLNZ+m+zVBb6VT7coZm91VJPhJGcoquJXT+zmvRDyc18Q61MUrUKAIYo11rVOOULvRTxJseQ1uPZ+UzpEuFwv0SkQKtVOrCrp1dH0WbT0bi6r23I9EzO2/5UkClcHTZ2OnWVBlQdqp6THvy+2cyrvUrVGq1HZ3c5mdtWJ/oOAA0AAA0mowlOVyMTywguylTp34TY2dgsxAsSTJcBgCTuv7B8b7po1cQqLlWxNrM1hbtC6bu3j3b/AFxifLz+Q5/jEjxlVVXq11A1JB0ZvDcOHtPGY1VweB8fyk1fFE+r7q/KU3rH6vur8ppui4MT7GVBfd4bzKzo3I+BkzuG0NgeB0A7jb7/ANCDKxJG62++lud5yez6UVQKljwzb9fRUes39B+QKZ/rj3PyjWbSw0XeebHmflw++POeBt3SG6Jus/8AsHuflAVyP9ovuD8MhLtzPiYnWN6zeJlstGhhNrVKbrUSuqsjB1PVIbEG43pKVQI7M7VhmZmdjkbUsbk2AsNTwjOub1m94xDXf1294yaFFrCYXDs6q+LWmpYBm6qo2UE6nKAL90pYmnkdkuTlZluVKkgEgEq2q332OokiYuopBWo6kG4Idgb+MTF4l6rtUqMXdzmZm9JjzMjBXhCEhQhCEAUQgIsqA+rVZ2LuzOxtdmJZjYWFydTYAD2RIixZQKIsIogCgRREiwB6mTI0gWSKYMMtI0sI8pI0mRpqMjhkgpI1cNiipBBsec0Vqo/pDKeYHmnvXh7PCeeV5YStOlpnz8mFpmnV2XfVbMOa6j28R7ZRfZ3ZJKeKI1BtLi7Rfic32gGP8QkcTnc49GUdn9hj02d2Hx/Kav8AaH1U90f0gdongFH7i/1EcC/LkKuH2ZfcjHuN/wCkuLhqaekbnkpDH2m1h+tJVrY9m0LEjlfTw3SrUxEKKM1OfbNDEY1bZQpVeQO/vNtfu7Jm1a6n9lveH4ZA9SQM0rlR3xYCZnT1H99fwSPFVKZRAiOrDPnLOHDXIy2AUWtr4yBmkLGcpOz3wgo9CNEdyRY7h4m26/O3wiNGGQ7IaTGmPjYNIQxsdEMFGmJHRsAbCK0SANMIpiTICEIogCwhATQHCKIkcIARwjY6AKI6NiiCMcBHCIIogyyQRytGCOEGWaOysC9eqtGmBma5uxyoqqLszN+yoAJJ/rL2Mw2ERHyYxqrqLgCgVpOeSsWuB9YrraO6G1V66rRZlQ4jDYnCI7GyLUqBcuY8ASuX94SjjdiYqkr9ZhqyBAS7FH6teF84GUjtBtM8vyqzDja6L+3dlHD16iIHdKYolnKkgF6SOQzKLDV/C0pUaNR1LIjuq+kVRmVe8gWE93/aNVtvLhjUY0itOk1O56oq2EDsGXcxJO8i+g5CZvRPDVKdPBVVOKq9bVYhaNQphaKpUVXat5rBiQCxBygqLXhZWo79I5ywRctGFhtm1noPiEXMiMiGwYsxa+qgCxAym5vpLFHZiLSWtiK4oLUBemioatZ0BtnKgqFUncSdbGwmw1SsKG0aWHarenjbKlIvdENSqGyqmqrprbTTWU9rYV8QuGxdCicRTWhh8PUp0wzFHpAhqbqnnqpFiGFtDw0u+RsnwRRSwOy6Vau6JiCaaUKmJaoaJDWQAuoplt+u/NFqbIodWuIXFk0C5oO/UN1tOpkzqppZvODAekGm/h8LTo4gn6N1QfZNfEVaJaovnHMHUlyWS4UDmJ5HaW189JcPTorh6Kua2RWd2LlcuZ3YksQugGgAMKcpPT/w0scYraL+1di4ejTR/pjO1Sl9IpL9HKhgSwUM3WHLcqRuNp5tmm/0oP8AdYH/ACVL+ZUnnTEW2ts3SXQjGNMVohmjaGNEjo2CjCIRxjDBoQxGixINDYhixDAEMbHRsAI2OhaQDYohCQCxViwmgEdCEAVYohCAPgIsIIOjhCEGGKI4QhBGKRJ2xdRkFM1HKDQIXY0x3KTaLCDLGis+bPnbNp52Y591vS37tI5K7qpRXdVJDFQzBCRuJUGxOg17IQgyxUxLqSyu6lrhirMGa+/MQbm/bEo13Q5kdkO66MVNuVxCEGRDVcksXYkggnMcxB3gneRGGEIKgZybXJNhYXJNhyHIdkYYQg0hDGwhBpCGNaEINCGMaLCDSEjDCEFQhiQhBRsQxYQBsIQkYP/Z', width=700)
# st.title('3. Video')
# video_file = open('/Users/ashishzangra/Documents/Streamlit/video.mp4', 'rb')
# st.video(video_file, start_time=20)
# st.title('4. Audio')
# audio_file = open('/Users/ashishzangra/Documents/Streamlit/sample.mp3', 'rb')
# st.audio(audio_file.read())




def convert_image(image_path, new_format):
    with Image.open(image_path) as img:

        new_name = image_path.name.split('.')[0] + '.' + new_format
        final_path =  new_name

        img = img.convert('RGB')

        st.subheader(final_path)
        img.save(final_path)
        st.success('Image Saved at ' + final_path)
        
st.title('Image Converter')

# File Uploader for Image
image_path = st.file_uploader('Upload your image', type=['png', 'jpg', 'jpeg'])

# Format Selection Dropdown
new_format = st.selectbox('Select the output format', ['png', 'jpeg', 'jpg'])

# Conversion Button
if st.button('Convert'):
    if image_path is not None:
        convert_image(image_path, new_format)
    else:
        st.error('Please upload the image file')
        



        
# st.title('Image Rotator')

# def rotate_image(image, angle):
#     img = np.array(image)
#     height, width = img.shape[:2]
#     M = cv.getRotationMatrix2D((width/2, height/2), angle, 1)
#     rotated_img = cv.warpAffine(img, M, (width, height))
#     return rotated_img



# st.subheader('Upload an image file: ')
# img_file = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])

# st.subheader('Rotate the Image:')
# angle = st.slider('Choose the Angle:', -180, 180, 0, 1)

# if img_file is not None:
#     image = Image.open(img_file)
#     rotated_img = rotate_image(image, angle)
#     st.image(rotated_img)
        