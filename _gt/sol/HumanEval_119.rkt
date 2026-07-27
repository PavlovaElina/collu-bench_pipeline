#lang racket
(define (match_parens lst)
  (define (good? s)
    (let loop ([cs (string->list s)] [d 0])
      (cond [(< d 0) #f]
            [(null? cs) (= d 0)]
            [(char=? (car cs) #\() (loop (cdr cs) (+ d 1))]
            [else (loop (cdr cs) (- d 1))])))
  (define a (string-append (first lst) (second lst)))
  (define b (string-append (second lst) (first lst)))
  (if (or (good? a) (good? b)) "Yes" "No"))
