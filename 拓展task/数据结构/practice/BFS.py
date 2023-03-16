from collections import deque


def search_seller(name):
    search_queue = deque()
    search_queue += graph[name]
    searched_list = []
    while search_queue:
        person = search_queue.popleft()
        if person not in searched_list:
            length += 1
            if person_is_seller(person):
                print(f"{person} is a seller")
                return True
            else:
                search_queue += graph[person]
                searched_list.append(person)
    print(searched_list)
    return False


def person_is_seller(person):
    return person[0] == 'a'


if __name__ == "__main__":
    name = "bob"
    graph = {
        "alice": ["linda", "john"],
        "bob": ["gai", 'jing'],
        "gai": ["alice"],
        "jing": [],
        "linda": [],
        "john": []
    }
    print(search_seller(name))
