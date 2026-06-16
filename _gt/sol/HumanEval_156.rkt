#lang racket
(define (int_to_mini_roman number)
  (define vals '((1000 . "m") (900 . "cm") (500 . "d") (400 . "cd") (100 . "c")
                 (90 . "xc") (50 . "l") (40 . "xl") (10 . "x") (9 . "ix")
                 (5 . "v") (4 . "iv") (1 . "i")))
  (let loop ([n number] [vs vals] [acc ""])
    (cond [(= n 0) acc]
          [(>= n (caar vs)) (loop (- n (caar vs)) vs (string-append acc (cdar vs)))]
          [else (loop n (cdr vs) acc)])))
